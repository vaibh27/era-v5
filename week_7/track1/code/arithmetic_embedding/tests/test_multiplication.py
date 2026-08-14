from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arithmetic.data.generators import local_product_batch
from arithmetic.models.addition import RecurrentLatentAdder
from arithmetic.models.multiplication import CompositionalLatentMultiplier, SchoolbookLatentMultiplier


def test_local_product_targets_are_correct():
    batch = local_product_batch(10, 100, random.Random(4), torch.device("cpu"))
    product = batch["a"] * batch["b"]
    assert torch.equal(batch["low"], product % 10)
    assert torch.equal(batch["high"], product // 10)


def test_multiplier_preserves_variable_output_shape():
    model = SchoolbookLatentMultiplier(base=10, limb_dim=8, hidden_dim=16)
    result = model(torch.tensor([[3, 2]]), torch.tensor([[5, 1]]))
    assert result.shape == (1, 5)


def test_compositional_multiplier_preserves_variable_output_shape():
    product = SchoolbookLatentMultiplier(base=10, limb_dim=8, hidden_dim=16)
    adder = RecurrentLatentAdder(base=10, limb_dim=8, hidden_dim=16)
    result = CompositionalLatentMultiplier(product, adder)(torch.tensor([[3, 2]]), torch.tensor([[5, 1]]))
    assert result.shape == (1, 5)
