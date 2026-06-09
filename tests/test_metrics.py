import numpy as np

from multitask_bert.evaluation.metrics import (
    accuracy,
    f1_macro,
    normalised_pearson,
    pearson_r,
)


def test_accuracy_perfect_and_zero():
    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy([1, 2, 3], [3, 2, 1]) == 1.0 / 3.0


def test_f1_macro_balanced():
    # 2-class perfect predictions -> F1 = 1
    f1 = f1_macro([0, 1, 0, 1], [0, 1, 0, 1], num_classes=2)
    assert f1 == 1.0


def test_pearson_perfect():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    assert abs(pearson_r(x, 2.0 * x + 3.0) - 1.0) < 1e-6


def test_normalised_pearson_range():
    n = normalised_pearson([0, 1, 2, 3], [0, 1, 2, 3])
    assert abs(n - 1.0) < 1e-6
    n = normalised_pearson([0, 1, 2, 3], [3, 2, 1, 0])
    assert abs(n - 0.0) < 1e-6
