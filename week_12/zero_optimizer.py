"""
Minimal, from-scratch simulation of ZeRO-1 / ZeRO-2 / ZeRO-3 sharding built directly
on torch.distributed collectives (no DeepSpeed). Stage 0 is plain data-parallel
(full replication) used as the baseline to compare against.

Sharding granularity: every parameter tensor is flattened, zero-padded to a
multiple of world_size, and cut into world_size equal chunks. This mirrors real
ZeRO's flat-buffer sharding (sub-tensor, not "whole tensor per rank").

Stage semantics:
  0 - baseline DP: every rank holds full params, full grads, full optimizer state.
      Grads are all-reduced (averaged) every step, like vanilla DDP.
  1 - ZeRO-1: params & grads stay full on every rank (grads all-reduced like DDP),
      but Adam's (m, v) state is sharded: each rank only updates its 1/N slice of
      each parameter, then the updated slices are all-gathered back into the full
      parameter tensor on every rank.
  2 - ZeRO-2: adds gradient sharding. Instead of all-reduce, gradients are
      reduce-scattered, so each rank only ever materializes its own grad shard.
      Params are still gathered back to full after each step (still fully
      replicated at rest, like stage 1).
  3 - ZeRO-3: adds parameter sharding. Only the local 1/N shard of each parameter
      is kept at rest. Immediately before a module needs its parameters it
      all-gathers them into a temporary full tensor; a backward hook on that
      temporary tensor reduce-scatters its gradient back down to a shard and the
      full tensor is dropped, so full-size params/grads only exist transiently
      during that module's forward/backward.
"""

import torch
import torch.distributed as dist

from shard_utils import (
    shard_size, flatten_pad, local_shard, all_gather_flat,
    reduce_scatter_flat, unpad_reshape,
)


class ZeroOptimizer:
    def __init__(self, model, stage, rank, world_size, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, pg=None):
        self.model = model
        self.stage = stage
        self.rank = rank
        self.world_size = world_size
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.pg = pg
        self.t = 0

        self.named_params = list(model.named_parameters())

        # persistent per-rank Adam state: full for stage 0, sharded from stage 1 upward
        self.m, self.v = {}, {}
        for name, p in self.named_params:
            n = shard_size(p.numel(), world_size) if stage >= 1 else p.numel()
            self.m[name] = torch.zeros(n)
            self.v[name] = torch.zeros(n)

        self._grad_shard = {}      # name -> local reduced grad shard (stage 2 & 3)
        self._persistent_shard = {}  # name -> local param shard storage (stage 3 only)
        self._owner_module = {}    # name -> (module, attr) for stage-3 hook plumbing

        if stage == 3:
            self._convert_to_stage3()

    # ---------------- stage 3 param -> shard conversion ----------------

    def _convert_to_stage3(self):
        name_to_owner = {}
        for mod_name, module in self.model.named_modules():
            for attr, p in list(module.named_parameters(recurse=False)):
                full_name = f"{mod_name}.{attr}" if mod_name else attr
                name_to_owner[full_name] = (module, attr)

        for name, p in self.named_params:
            module, attr = name_to_owner[name]
            shard = local_shard(p.data, self.rank, self.world_size)
            self._persistent_shard[name] = shard
            self._owner_module[name] = (module, attr, p.shape, p.numel())
            del module._parameters[attr]
            setattr(module, attr, shard.clone())  # placeholder until first gather

    def gather_full_params(self):
        """Call once before each forward pass. All-gathers every parameter's
        shard into a full tensor and installs it on its owning module, so any
        internal attribute access (even ones that bypass submodule __call__,
        e.g. nn.MultiheadAttention reading self.out_proj.weight directly) sees
        a correctly-shaped tensor. A backward hook on each full tensor
        reduce-scatters its gradient and restores the small shard afterward,
        so at-rest memory (between steps) still only holds the shard."""
        for name, (module, attr, shape, numel) in self._owner_module.items():
            shard = self._persistent_shard[name]
            full_flat = all_gather_flat(shard, self.world_size, self.pg)
            full = unpad_reshape(full_flat, shape).clone().requires_grad_(True)
            full.register_hook(self._make_grad_hook(name))
            setattr(module, attr, full)

    def _make_grad_hook(self, name):
        def grad_hook(grad):
            module, attr, shape, numel = self._owner_module[name]
            padded = flatten_pad(grad, self.world_size)
            self._grad_shard[name] = reduce_scatter_flat(padded, self.world_size, self.pg)
            # drop the full-size param now that its shard of grad is saved, so
            # memory returns to shard-only "at rest" size immediately, not just
            # at the next forward's gather.
            setattr(module, attr, self._persistent_shard[name])
            return grad
        return grad_hook

    # ---------------- grad sync (stage 0 / 1 / 2) ----------------

    def sync_grads(self):
        """Call after loss.backward(). Stage 3 grads are already handled by hooks."""
        if self.stage == 3:
            return
        for name, p in self.named_params:
            if p.grad is None:
                continue
            if self.stage in (0, 1):
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.pg)
                p.grad /= self.world_size
            else:  # stage 2: reduce-scatter, only keep this rank's shard
                padded = flatten_pad(p.grad, self.world_size)
                self._grad_shard[name] = reduce_scatter_flat(padded, self.world_size, self.pg)
                p.grad = None  # free the full-size grad now that the shard is saved

    # ---------------- optimizer step ----------------

    def _adam(self, name, param_slice, grad_slice):
        b1, b2 = self.betas
        m, v = self.m[name], self.v[name]
        m.mul_(b1).add_(grad_slice, alpha=1 - b1)
        v.mul_(b2).addcmul_(grad_slice, grad_slice, value=1 - b2)
        m_hat = m / (1 - b1 ** self.t)
        v_hat = v / (1 - b2 ** self.t)
        param_slice.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def step(self):
        self.t += 1
        if self.stage == 0:
            for name, p in self.named_params:
                self._adam(name, p.data.reshape(-1)[:p.numel()], p.grad.reshape(-1))
            self.zero_grad()
            return

        if self.stage in (1, 2):
            for name, p in self.named_params:
                s = shard_size(p.numel(), self.world_size)
                shard = local_shard(p.data, self.rank, self.world_size)
                if self.stage == 1:
                    grad_padded = flatten_pad(p.grad, self.world_size)
                    grad_shard = grad_padded[self.rank * s:(self.rank + 1) * s]
                else:
                    grad_shard = self._grad_shard[name]
                self._adam(name, shard, grad_shard)
                full_flat = all_gather_flat(shard, self.world_size, self.pg)
                p.data.copy_(unpad_reshape(full_flat, p.shape))
            self.zero_grad()
            self._grad_shard.clear()
            return

        # stage 3: update the persistent local shard directly, no gather needed at rest
        for name, p in self.named_params:
            shard = self._persistent_shard[name]
            grad_shard = self._grad_shard.get(name)
            if grad_shard is None:
                continue
            self._adam(name, shard, grad_shard)
        self._grad_shard.clear()

    def zero_grad(self):
        for _, p in self.named_params:
            p.grad = None

    # ---------------- memory accounting ----------------

    def memory_report(self, dtype_bytes=4):
        """Theoretical steady-state per-rank bytes for one training step:
        params, the grad buffer produced by backward, and Adam's (m, v).
        Stage 0/1 keep grads full (only stage 1's optimizer state is sharded);
        stage 2/3 additionally shard the grad buffer; stage 3 also shards params."""
        params_bytes = grads_bytes = optim_bytes = 0
        for name, p in self.named_params:
            full = p.numel() * dtype_bytes
            s = shard_size(p.numel(), self.world_size) * dtype_bytes
            params_bytes += s if self.stage == 3 else full
            grads_bytes += s if self.stage >= 2 else full
            optim_bytes += 2 * s if self.stage >= 1 else 2 * full
        return {
            "params_MB": params_bytes / 1e6,
            "grads_MB": grads_bytes / 1e6,
            "optim_MB": optim_bytes / 1e6,
            "total_MB": (params_bytes + grads_bytes + optim_bytes) / 1e6,
        }
