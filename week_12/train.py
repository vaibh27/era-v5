import argparse
import gc
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from model import DemoLM, make_batch
from zero_optimizer import ZeroOptimizer


def raise_fd_limit(target=4096):
    """32 gloo ranks doing full-mesh collectives open far more sockets than
    macOS's default per-process file-descriptor limit (often 256), which
    corrupts gloo's `uv` transport under load ('Unexpected opcode' errors,
    not a real protocol bug). Raise it defensively in both the parent
    (before spawning) and each spawned child, since spawned processes don't
    reliably inherit a limit raised only in the parent."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(target, hard) if hard > 0 else target
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ImportError, ValueError, OSError):
        pass  # not on POSIX, or not permitted to raise it here


raise_fd_limit()


class Args:
    """Plain config bag. Defined here (not inline in a script/notebook) so it
    has a real module path and torch.multiprocessing.spawn can pickle it to
    send to child processes."""

    def __init__(self, stage, batch_size=4, seq_len=32, d_model=256,
                 n_layers=4, n_heads=4, vocab_size=2000, steps=5):
        self.stage = stage
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.steps = steps


def broadcast_init(model, pg=None):
    for p in model.parameters():
        dist.broadcast(p.data, src=0, group=pg)


def worker(rank, world_size, args, port, results):
    raise_fd_limit()
    torch.set_num_threads(1)
    torch.manual_seed(0)  # same init across ranks, then broadcast to be sure

    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )

    model = DemoLM(
        vocab_size=args.vocab_size, seq_len=args.seq_len, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
    )
    broadcast_init(model)
    total_params = sum(p.numel() for p in model.parameters())

    opt = ZeroOptimizer(model, stage=args.stage, rank=rank, world_size=world_size, lr=1e-3)

    step_times = []
    losses = []
    dist.barrier()
    for step in range(args.steps):
        t0 = time.perf_counter()
        x, y = make_batch(args.batch_size, args.seq_len, args.vocab_size, seed=rank * 10_000 + step)
        if args.stage == 3:
            opt.gather_full_params()
        out = model(x)
        loss = F.cross_entropy(out.reshape(-1, args.vocab_size), y.reshape(-1))
        loss.backward()
        opt.sync_grads()
        opt.step()
        dist.barrier()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())

    gc.collect()
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
    mem = opt.memory_report()

    if rank == 0:
        results["mem"] = mem
        results["params"] = total_params
    results[f"rss_mb_{rank}"] = rss_mb
    results[f"step_time_s_{rank}"] = sum(step_times) / len(step_times)
    results[f"final_loss_{rank}"] = losses[-1]

    dist.barrier()
    dist.destroy_process_group()


def run(stage, world_size, args, port):
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(worker, args=(world_size, args, port, results), nprocs=world_size, join=True)
    flat = dict(results)
    return {
        "mem": flat["mem"],
        "params": flat["params"],
        "rss_mb": {r: flat[f"rss_mb_{r}"] for r in range(world_size)},
        "step_time_s": {r: flat[f"step_time_s_{r}"] for r in range(world_size)},
        "final_loss": {r: flat[f"final_loss_{r}"] for r in range(world_size)},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--port", type=int, default=29500)
    args = ap.parse_args()

    res = run(args.stage, args.world_size, args, args.port)
    rss = res["rss_mb"]
    times = res["step_time_s"]
    print(f"stage={args.stage} world_size={args.world_size} params={res['params']:,}")
    print(f"  theoretical/rank: {res['mem']}")
    print(f"  actual RSS/rank MB: min={min(rss.values()):.1f} max={max(rss.values()):.1f} "
          f"mean={sum(rss.values()) / len(rss):.1f}")
    print(f"  step time s: min={min(times.values()):.4f} max={max(times.values()):.4f} "
          f"mean={sum(times.values()) / len(times):.4f}")
    print(f"  final losses (rank0): {res['final_loss'][0]:.3f}")
