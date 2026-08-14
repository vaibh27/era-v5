from __future__ import annotations

import random

import torch

from arithmetic.data.generators import addition_batch
from arithmetic.models.addition import RecurrentLatentAdder
from arithmetic.reference.limbs import limbs_to_int


@torch.no_grad()
def addition_metrics(model: RecurrentLatentAdder, base: int, min_limbs: int, max_limbs: int, examples: int, batch_size: int, seed: int, device: torch.device) -> dict[str, float | int]:
    model.eval()
    rng = random.Random(seed)
    exact = limbs = carries = total = 0
    for start in range(0, examples, batch_size):
        target = addition_batch(base, min_limbs, max_limbs, min(batch_size, examples - start), rng, device)
        output = model(target["a"], target["b"])
        predicted_limbs = output.result_logits.argmax(dim=-1)
        predicted_carries = output.carry_logits.argmax(dim=-1)
        limbs += int((predicted_limbs == target["result"]).sum().item())
        carries += int((predicted_carries == target["carry"]).sum().item())
        total += target["result"].numel()
        for predicted, expected in zip(predicted_limbs.cpu().tolist(), target["result"].cpu().tolist()):
            exact += int(limbs_to_int(predicted, base) == limbs_to_int(expected, base))
    return {
        "examples": examples,
        "limb_exact_accuracy": limbs / total,
        "carry_exact_accuracy": carries / total,
        "integer_exact_accuracy": exact / examples,
    }


def length_generalization(model: RecurrentLatentAdder, base: int, train_max_limbs: int, test_lengths: list[int], examples: int, batch_size: int, seed: int, device: torch.device) -> dict[str, dict[str, float | int]]:
    return {
        str(length): addition_metrics(model, base, length, length, examples, batch_size, seed + length, device)
        for length in test_lengths
    }
