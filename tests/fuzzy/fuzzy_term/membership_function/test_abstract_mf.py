import copy

import numpy as np
import pytest

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf import AbstractMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.dont_care_mf import DontCareMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.rectangular_mf import RectangularMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF


def test_empty_params():
    mf = AbstractMF(np.empty(0))
    assert len(mf.get_params()) == 0


def test_none_params():
    mf = AbstractMF(None)
    assert len(mf.get_params()) == 0


def test_set_param_value_none_index():
    with pytest.raises(Exception):
        mf = AbstractMF(np.array([0.0, 1.0, 2.0]))
        mf.set_param_value(None, 4)


def test_eq_true_dc():
    mf1 = DontCareMF()
    mf2 = DontCareMF()
    assert mf1 == mf2


def test_eq_none():
    mf1 = TriangularMF(0, 0.5, 1)
    assert mf1 != None


def test_eq_true_triangular():
    mf1 = TriangularMF(0, 0.5, 1)
    mf2 = TriangularMF(0, 0.5, 1)
    assert mf1 == mf2


def test_eq_true_rectangular():
    mf1 = RectangularMF(0, 0.5)
    mf2 = RectangularMF(0, 0.5)
    assert mf1 == mf2


def test_eq_different_function():
    mf1 = DontCareMF()
    mf2 = TriangularMF()
    assert mf1 != mf2
