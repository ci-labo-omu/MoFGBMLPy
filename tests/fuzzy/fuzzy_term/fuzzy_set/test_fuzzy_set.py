import copy

import numpy as np
import pytest
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set import FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.dont_care_mf import DontCareMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.rectangular_mf import RectangularMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.rectangular_fuzzy_set import RectangularFuzzySet


# def test_none_function():
#     with pytest.raises(Exception):
#         _ = FuzzySet(None, 0, DivisionType.EQUAL_DIVISION, "term")
#
#
# def test_none_id():
#     with pytest.raises(TypeError):
#         _ = FuzzySet(DontCareMF(), None, DivisionType.EQUAL_DIVISION, "term")
#
#
# def test_none_division_type():
#     with pytest.raises(TypeError):
#         _ = FuzzySet(DontCareMF(), 0, None, "term")
#
#
# def test_none_term():
#     with pytest.raises(TypeError):
#         _ = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION, None)
#
#
# def test_get_division_type():
#     fs = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
#     assert fs.get_division_type() == DivisionType.EQUAL_DIVISION
#
#
# def test_eq_true_dc():
#     fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
#     fs2 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
#     assert fs1 == fs2
#
#
# def test_eq_true_triangular():
#     fs1 = FuzzySet(TriangularMF(0, 0.5, 1), 0, DivisionType.EQUAL_DIVISION)
#     fs2 = FuzzySet(TriangularMF(0, 0.5, 1), 0, DivisionType.EQUAL_DIVISION)
#     assert fs1 == fs2
#
#
# def test_eq_true_rectangular():
#     fs1 = FuzzySet(RectangularMF(0, 0.5), 0, DivisionType.EQUAL_DIVISION)
#     fs2 = FuzzySet(RectangularMF(0, 0.5), 0, DivisionType.EQUAL_DIVISION)
#     assert fs1 == fs2
#
#
# def test_eq_different_function():
#     fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
#     fs2 = FuzzySet(TriangularMF(), 0, DivisionType.EQUAL_DIVISION)
#     assert fs1 != fs2
#
#
# def test_eq_different_id():
#     fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
#     fs2 = FuzzySet(DontCareMF(), 1, DivisionType.EQUAL_DIVISION)
#     assert fs1 != fs2
#
#
# def test_eq_different_division_type():
#     fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION)
#     fs2 = FuzzySet(DontCareMF(), 0, DivisionType.ENTROPY_DIVISION)
#     assert fs1 != fs2
#
#
# def test_eq_different_term():
#     fs1 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION, "term1")
#     fs2 = FuzzySet(DontCareMF(), 0, DivisionType.EQUAL_DIVISION, "term2")
#     assert fs1 != fs2
#
#
# def test_deep_copy():
#     fs = FuzzySet(TriangularMF(1.0, 2.0, 3.0), 0, DivisionType.EQUAL_DIVISION)
#     fs_copy = copy.deepcopy(fs)
#
#     assert fs == fs_copy


def test_none_id():
    with pytest.raises(TypeError):
        _ = DontCareFuzzySet(None)

def test_none_term():
    with pytest.raises(TypeError):
        _ = TriangularFuzzySet(0.1, 0.2, 0.3, 0, None)


def test_get_division_type():
    fs = DontCareFuzzySet(0)
    assert fs.get_division_type() == DivisionType.EQUAL_DIVISION


def test_eq_true_dc():
    fs1 = DontCareFuzzySet(0)
    fs2 = DontCareFuzzySet(0)
    assert fs1 == fs2


def test_eq_true_triangular():
    fs1 = TriangularFuzzySet(0, 0.5, 1, 0, "medium")
    fs2 = TriangularFuzzySet(0, 0.5, 1, 0, "medium")
    assert fs1 == fs2


def test_eq_true_rectangular():
    fs1 = RectangularFuzzySet(0, 0.5, 0, "medium")
    fs2 = RectangularFuzzySet(0, 0.5, 0, "medium")
    assert fs1 == fs2


def test_eq_different_function():
    fs1 = DontCareFuzzySet(0)
    fs2 = TriangularFuzzySet(0.1, 0.4, 0.6, 0, "medium")
    assert fs1 != fs2


def test_eq_different_id():
    fs1 = DontCareFuzzySet(0)
    fs2 = DontCareFuzzySet(1)
    assert fs1 != fs2


def test_eq_different_term():
    fs1 = TriangularFuzzySet(0.1, 0.4, 0.6, 0, "medium")
    fs2 = TriangularFuzzySet(0.1, 0.4, 0.6, 0, "large")
    assert fs1 != fs2


def test_deep_copy():
    fs = TriangularFuzzySet(0.1, 0.4, 0.6, 0, "medium")
    fs_copy = copy.deepcopy(fs)

    assert fs == fs_copy
