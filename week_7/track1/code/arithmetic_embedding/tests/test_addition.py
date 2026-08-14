from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arithmetic.data.generators import addition_batch, exhaustive_local_addition_batch, local_addition_batch, local_subtraction_batch
from arithmetic.models.subtraction import RecurrentLatentSubtractor
from arithmetic.models.addition import RecurrentLatentAdder
from arithmetic.reference.limbs import limbs_to_int


def test_batch_carries_and_result_reconstruct_exact_sum():
    batch = addition_batch(1000, 2, 2, 32, random.Random(3), torch.device("cpu"))
    for a, b, result in zip(batch["a"].tolist(), batch["b"].tolist(), batch["result"].tolist()):
        assert limbs_to_int(a, 1000) + limbs_to_int(b, 1000) == limbs_to_int(result, 1000)


def test_adder_uses_one_shared_cell_for_arbitrary_sequence_length():
    model = RecurrentLatentAdder(base=10, limb_dim=8, hidden_dim=16)
    a = torch.tensor([[9, 9, 0], [1, 2, 3]])
    b = torch.tensor([[1, 0, 0], [4, 5, 6]])
    output = model(a, b)
    assert output.result_logits.shape == (2, 3, 10)
    assert output.carry_logits.shape == (2, 3, 2)
    assert output.result_latents.shape == (2, 3, 8)


def test_local_transition_targets_match_base_arithmetic():
    batch = local_addition_batch(10, 128, random.Random(5), torch.device("cpu"))
    total = batch["a"] + batch["b"] + batch["carry_in"]
    assert torch.equal(batch["result"], total % 10)
    assert torch.equal(batch["carry"], total // 10)


def test_exhaustive_local_transitions_cover_every_operand_and_carry():
    batch = exhaustive_local_addition_batch(10, torch.device("cpu"))
    assert batch["a"].numel() == 2 * 10 * 10
    total = batch["a"] + batch["b"] + batch["carry_in"]
    assert torch.equal(batch["result"], total % 10)
    assert torch.equal(batch["carry"], total // 10)


def test_exhaustive_batch_can_weight_maximal_sum_boundary_case():
    batch = exhaustive_local_addition_batch(10, torch.device("cpu"), hard_case_repeat=3)
    assert batch["a"].numel() == 2 * 10 * 10 + 3
    assert torch.equal(batch["a"][-3:], torch.tensor([9, 9, 9]))
    assert torch.equal(batch["result"][-3:], torch.tensor([9, 9, 9]))


def test_local_subtraction_targets_and_shared_borrow_model_shape():
    batch = local_subtraction_batch(10, 128, random.Random(9), torch.device("cpu"))
    total = batch["a"] - batch["b"] - batch["borrow_in"]
    assert torch.equal(batch["result"], total % 10)
    assert torch.equal(batch["borrow"], (total < 0).long())
    model = RecurrentLatentSubtractor(base=10, limb_dim=8, hidden_dim=16)
    output = model(torch.tensor([[0, 1]]), torch.tensor([[1, 0]]))
    assert output.result_logits.shape == (1, 2, 10)
    assert output.borrow_logits.shape == (1, 2, 2)
