"""Sanity checks on SMART helper functions."""

import torch

from multitask_bert.losses.smart import squared_error, symmetric_kl


def test_symmetric_kl_zero_for_identical():
    logits = torch.randn(8, 5)
    assert symmetric_kl(logits, logits).item() < 1e-6


def test_symmetric_kl_positive():
    a = torch.randn(8, 5)
    b = torch.randn(8, 5)
    assert symmetric_kl(a, b).item() > 0.0


def test_squared_error():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    assert squared_error(a, b).item() == 0.0
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    assert abs(squared_error(a, b).item() - 4.0) < 1e-6
