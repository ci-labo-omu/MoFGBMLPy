import copy

import numpy as np
import pytest

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF


def test_none_left():
    with pytest.raises(TypeError):
        TriangularMF(None, 0.5, 1)


def test_none_center():
    with pytest.raises(TypeError):
        TriangularMF(0, None, 1)


def test_none_right():
    with pytest.raises(TypeError):
        TriangularMF(0, 0.5, None)


def test_invalid_center_left():
    with pytest.raises(ValueError):
        TriangularMF(0, -1, 1)
  
    
def test_invalid_center_right():
    with pytest.raises(ValueError):
        TriangularMF(0, 2, 1)


def test_invalid_left_center():
    with pytest.raises(ValueError):
        TriangularMF(0.7, 0.5, 1)


def test_invalid_left_right():
    with pytest.raises(ValueError):
        TriangularMF(2, 0.5, 1)
    
    
def test_invalid_right_left():
    with pytest.raises(ValueError):
        TriangularMF(0, 0.5, -1)


def test_invalid_right_center():
    with pytest.raises(ValueError):
        TriangularMF(0, 0.5, 0.2)


@pytest.mark.parametrize("x", np.concatenate([np.array([-0.5, 0.5, 1.1]), np.random.uniform(low=-1, high=2, size=(10,))]))
def test_get_value_different_params(x):
    left = -0.5
    center = 0.5
    right = 1.1
    mf = TriangularMF(left, center, right)
    precision = 1e-6

    if x <= left:
        assert abs(mf.get_value_py(x)) < precision
    elif x <= center:
        assert abs(mf.get_value_py(x) - ((x - left) / (center - left))) < precision
    elif x <= right:
        assert abs(mf.get_value_py(x) - ((right - x) / (right - center)) < precision)
    else:
        assert abs(mf.get_value_py(x)) < precision


@pytest.mark.parametrize("x", np.concatenate([np.array([0, 1]), np.random.uniform(low=-1, high=2, size=(10,))]))
def test_get_value_same_left_center(x):
    left = 0
    center = left
    right = 1
    mf = TriangularMF(left, center, right)
    precision = 1e-6

    if x < left:
        assert abs(mf.get_value_py(x)) < precision
    elif x == left:
        assert abs(mf.get_value_py(x) - 1) < precision
    elif x <= right:
        assert abs(mf.get_value_py(x) - ((right - x) / (right - center)) < precision)
    else:
        assert abs(mf.get_value_py(x)) < precision


@pytest.mark.parametrize("x", np.concatenate([np.array([0, 1]), np.random.uniform(low=-1, high=2, size=(10,))]))
def test_get_value_same_left_center(x):
    left = 0
    center = 1
    right = center
    mf = TriangularMF(left, center, right)
    precision = 1e-6

    if x < left:
        assert abs(mf.get_value_py(x)) < precision
    elif x < center:
        assert abs(mf.get_value_py(x) - ((x - left) / (center - left))) < precision
    elif x == center:
        assert abs(mf.get_value_py(x) - 1) < precision
    else:
        assert abs(mf.get_value_py(x)) < precision


@pytest.mark.parametrize("x", np.concatenate([np.array([0]), np.random.uniform(low=-1, high=2, size=(10,))]))
def test_get_value_all_equal(x):
    left = 0
    center = left
    right = center
    mf = TriangularMF(left, center, right)
    precision = 1e-6

    if x == center:
        assert abs(mf.get_value_py(x) - 1) < precision
    else:
        assert abs(mf.get_value_py(x)) < precision


@pytest.mark.parametrize(("x_min", "x_max"), np.random.uniform(low=-1, high=2, size=(5, 2)))
def test_get_param_range_left(x_min, x_max):
    left = 0
    center = 0.5
    right = 1

    mf = TriangularMF(left, center, right)

    if x_min > left or x_max < right:
        with pytest.raises(ValueError):
            _ = mf.get_param_range(0, x_min, x_max)
    else:
        param_range = mf.get_param_range(0, x_min, x_max)
        assert param_range[0] == x_min and param_range[1] == center


@pytest.mark.parametrize(("x_min", "x_max"), np.random.uniform(low=-1, high=2, size=(5, 2)))
def test_get_param_range_center(x_min, x_max):
    left = 0
    center = 0.5
    right = 1

    mf = TriangularMF(left, center, right)

    if x_min > left or x_max < right:
        with pytest.raises(ValueError):
            _ = mf.get_param_range(1, x_min, x_max)
    else:
        param_range = mf.get_param_range(1, x_min, x_max)
        assert param_range[0] == left and param_range[1] == right


@pytest.mark.parametrize(("x_min", "x_max"), np.random.uniform(low=-1, high=2, size=(5, 2)))
def test_get_param_range_right(x_min, x_max):
    left = 0
    center = 0.5
    right = 1

    mf = TriangularMF(left, center, right)

    if x_min > left or x_max < right:
        with pytest.raises(ValueError):
            _ = mf.get_param_range(2, x_min, x_max)
    else:
        param_range = mf.get_param_range(2, x_min, x_max)
        assert param_range[0] == center and param_range[1] == x_max


def test_get_plot_points():
    left = 0
    center = 0.5
    right = 1

    mf = TriangularMF(left, center, right)
    plot_points = mf.get_plot_points()
    assert np.array_equal(plot_points, np.array([[0,0], [left,0], [center,1], [right,0], [1,0]], dtype=np.float64))


def test_eq_true_dc():
    mf1 = TriangularMF(0, 0.5, 1)
    mf2 = TriangularMF(0, 0.5, 1)
    assert mf1 == mf2


def test_eq_different_left():
    mf1 = TriangularMF(0, 0.5, 1)
    mf2 = TriangularMF(0.1, 0.5, 1)
    assert mf1 != mf2


def test_eq_different_center():
    mf1 = TriangularMF(0, 0.5, 1)
    mf2 = TriangularMF(0, 0.6, 1)
    assert mf1 != mf2


def test_eq_different_right():
    mf1 = TriangularMF(0, 0.5, 1)
    mf2 = TriangularMF(0, 0.5, 2)
    assert mf1 != mf2


def test_deep_copy():
    mf = TriangularMF(0, 0.5, 1)
    mf_copy = copy.deepcopy(mf)
    assert mf == mf_copy and id(mf.get_params()) != id(mf_copy.get_params())
