from __future__ import annotations

import torch
import torch.nn.functional as F

from arithmetic.models.addition import AdditionOutput, RecurrentLatentAdder


def addition_loss(model: RecurrentLatentAdder, output: AdditionOutput, target: dict[str, torch.Tensor], latent_weight: float = 0.1) -> dict[str, torch.Tensor]:
    """Supervise both observable limbs and the internal carry/latent states."""
    limb = F.cross_entropy(output.result_logits.flatten(0, 1), target["result"].flatten())
    carry = F.cross_entropy(output.carry_logits.flatten(0, 1), target["carry"].flatten())
    target_latents = model.embedding(target["result"]).detach()
    # Compare directions so the classifier cannot satisfy this auxiliary term by
    # merely scaling the learned embedding table.
    latent = F.mse_loss(F.normalize(output.result_latents, dim=-1), F.normalize(target_latents, dim=-1))
    total = limb + carry + latent_weight * latent
    return {"total": total, "limb": limb, "carry": carry, "latent": latent}
