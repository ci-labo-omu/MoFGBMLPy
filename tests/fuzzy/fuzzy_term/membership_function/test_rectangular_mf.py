import copy

import numpy as np
import pytest

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.rectangular_mf import RectangularMF


def test_none_left():
    with pytest.raises(TypeError):
        RectangularMF(None, 1)


def test_none_right():
    with pytest.raises(TypeError):
        RectangularMF(0, None)


def test_invalid_left_right():
    with pytest.raises(ValueError):
        RectangularMF(2, 1)


@pytest.mark.parametrize("x", np.concatenate([np.array([-0.5, 1.1]), np.random.uniform(low=-1, high=2, size=(10,))]))
def test_get_value_different_params(x):
    left = -0.5
    right = 1.1
    mf = RectangularMF(left, right)
    precision = 1e-6

    if x < left or x > right:
        assert abs(mf.get_value_py(x)) < precision
    else:
        assert abs(mf.get_value_py(x) - 1) < precision

@pytest.mark.parametrize("x", np.concatenate([np.array([0]), np.random.uniform(low=-1, high=1, size=(5,))]))
def test_get_value_all_equal(x):
    left = 0
    right = 0
    mf = RectangularMF(left, right)
    precision = 1e-6

    if x == left:
        assert abs(mf.get_value_py(x) - 1) < precision
    else:
        assert abs(mf.get_value_py(x)) < precision


@pytest.mark.parametrize(("x_min", "x_max"), np.random.uniform(low=-1, high=2, size=(5, 2)))
def test_get_param_range_left(x_min, x_max):
    left = 0
    right = 1

    mf = RectangularMF(left, right)

    if x_min > left or x_max < right:
        with pytest.raises(ValueError):
            _ = mf.get_param_range(0, x_min, x_max)
    else:
        param_range = mf.get_param_range(0, x_min, x_max)
        assert param_range[0] == x_min and param_range[1] == right


@pytest.mark.parametrize(("x_min", "x_max"), np.random.uniform(low=-1, high=2, size=(5, 2)))
def test_get_param_range_right(x_min, x_max):
    left = 0
    right = 1

    mf = RectangularMF(left, right)

    if x_min > left or x_max < right:
        with pytest.raises(ValueError):
            _ = mf.get_param_range(1, x_min, x_max)
    else:
        param_range = mf.get_param_range(1, x_min, x_max)
        assert param_range[0] == left and param_range[1] == x_max


def test_get_plot_points():
    left = 0
    right = 1

    mf = RectangularMF(left, right)
    plot_points = mf.get_plot_points()
    assert np.array_equal(plot_points, np.array([[0,0], [left,1], [right,1], [1,0]], dtype=np.float64))


def test_eq_true_dc():
    mf1 = RectangularMF(0, 1)
    mf2 = RectangularMF(0, 1)
    assert mf1 == mf2


def test_eq_different_left():
    mf1 = RectangularMF(0, 1)
    mf2 = RectangularMF(0.1, 1)
    assert mf1 != mf2


def test_eq_different_right():
    mf1 = RectangularMF(0, 1)
    mf2 = RectangularMF(0, 2)
    assert mf1 != mf2


def test_deep_copy():
    mf = RectangularMF(0, 1)
    mf_copy = copy.deepcopy(mf)
    assert mf == mf_copy and id(mf.get_params()) != id(mf_copy.get_params())
