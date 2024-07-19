import copy

import numpy as np
import pytest

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.dont_care_mf import DontCareMF


def test_get_param_range():
    mf = DontCareMF()
    assert len(mf.get_param_range(0)) == 0


def test_is_param_value_valid():
    mf = DontCareMF()
    assert not mf.is_param_value_valid(0, 0)


def test_get_plot_points():
    mf = DontCareMF()
    plot_points = mf.get_plot_points()
    assert np.array_equal(plot_points, np.array([[0, 1], [1, 1]], dtype=np.float64))


def test_eq_true_dc():
    mf1 = DontCareMF()
    mf2 = DontCareMF()
    assert mf1 == mf2


def test_deep_copy():
    mf = DontCareMF()
    mf_copy = copy.deepcopy(mf)
    assert mf == mf_copy and id(mf.get_params()) != id(mf_copy.get_params())
