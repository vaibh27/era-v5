from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from continuous_crt import ContinuousCRTEmbedding, HybridArithmeticEmbedding


def test_continuous_crt_is_exact_for_all_pairs_in_z_105():
    model = ContinuousCRTEmbedding((3, 5, 7))
    numbers = torch.arange(model.system.capacity)
    embeddings = model.encode(numbers)
    assert model.dim == 15
    assert torch.equal(model.decode(embeddings), numbers)

    for x in range(model.system.capacity):
        ex = embeddings[x].expand(model.system.capacity, -1)
        assert torch.equal(model.decode(model.add(ex, embeddings)), (x + numbers) % model.system.capacity)
        assert torch.equal(model.decode(model.mul(ex, embeddings)), (x * numbers) % model.system.capacity)


def test_hybrid_embedding_preserves_math_prefix():
    model = HybridArithmeticEmbedding((3, 5, 7), semantic_vocab_size=10, semantic_dim=4)
    combined = model(torch.tensor([9]), torch.tensor([2]))
    assert combined.shape == (1, 19)
    assert model.math.decode(combined[:, : model.math.dim]).item() == 9
