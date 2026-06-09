"""The Hadamard product of normalized vectors should sum to cosine."""

import torch
import torch.nn.functional as F

from multitask_bert.models.relational import RichRelationalLayer


def test_hadamard_sum_equals_cosine():
    rng = torch.Generator().manual_seed(0)
    h = 16
    layer = RichRelationalLayer(h)
    a = torch.randn(4, h, generator=rng)
    b = torch.randn(4, h, generator=rng)

    expected = F.cosine_similarity(a, b, dim=-1)
    actual = layer.hadamard(a, b).sum(-1)
    assert torch.allclose(expected, actual, atol=1e-5)
