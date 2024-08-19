import copy

import numpy as np
import pytest
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set import FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.dont_care_mf import DontCareMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.rectangular_mf import RectangularMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF


def test_none_function():
    with pytest.raises(TypeError):
        _ = FuzzySet(None, 0, DivisionType.EQUAL_DIVISION, "term")


def test_none_id():
    with pytest.raises(TypeError):
        _ = FuzzySet(DontCareMF(), None, DivisionType.EQUAL_DIVISION, "term")


def test_none_division_type():
    with pytest.raises(TypeError):
        _ = FuzzySet(DontCareMF(), 0, None, "term")


def test_none_term():
    with pytest.raises(TypeError):
        _ = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION, None)


def test_get_division_type():
    fs = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
    assert fs.get_division_type() == DivisionType.EQUAL_DIVISION


def test_eq_true_dc():
    fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
    fs2 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
    assert fs1 == fs2


def test_eq_true_triangular():
    fs1 = FuzzySet(TriangularMF(0, 0.5, 1), 0, DivisionType.EQUAL_DIVISION)
    fs2 = FuzzySet(TriangularMF(0, 0.5, 1), 0, DivisionType.EQUAL_DIVISION)
    assert fs1 == fs2


def test_eq_true_rectangular():
    fs1 = FuzzySet(RectangularMF(0, 0.5), 0, DivisionType.EQUAL_DIVISION)
    fs2 = FuzzySet(RectangularMF(0, 0.5), 0, DivisionType.EQUAL_DIVISION)
    assert fs1 == fs2


def test_eq_different_function():
    fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
    fs2 = FuzzySet(TriangularMF(), 0, DivisionType.EQUAL_DIVISION)
    assert fs1 != fs2


def test_eq_different_id():
    fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
    fs2 = FuzzySet(DontCareMF(), 1, DivisionType.EQUAL_DIVISION)
    assert fs1 != fs2


def test_eq_different_division_type():
    fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
    fs2 = FuzzySet(DontCareMF(), 0, DivisionType.ENTROPY_DIVISION)
    assert fs1 != fs2


def test_eq_different_term():
    fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION, "term1")
    fs2 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION, "term2")
    assert fs1 != fs2


def test_deep_copy():
    fs = FuzzySet(TriangularMF(1.0,2.0,3.0), 0, DivisionType.EQUAL_DIVISION)
    fs_copy = copy.deepcopy(fs)

    assert fs == fs_copy
