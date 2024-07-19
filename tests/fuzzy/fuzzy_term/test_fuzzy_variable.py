import copy

import numpy as np
import pytest

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable


def test_none_name():
    with pytest.raises(Exception):
        _ = FuzzyVariable(name=None)


def test_none_fuzzy_sets():
    var = FuzzyVariable(fuzzy_sets=None, support_values=np.empty(0, dtype=np.float64))
    assert len(var.get_fuzzy_sets()) == 0


def test_none_support_values():
    var = FuzzyVariable(fuzzy_sets=np.empty(0, dtype=object))
    assert len(var.get_fuzzy_sets()) == 0


def test_different_support_fuzzy_sets_sizes():
    with pytest.raises(Exception):
        _ = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]), support_values=np.array([0.5, 1]))


def test_invalid_domain_size_greater():
    with pytest.raises(Exception):
        FuzzyVariable(domain=[0, 1, 2])


def test_invalid_domain_size_smaller():
    with pytest.raises(Exception):
        FuzzyVariable(domain=[0])


def test_invalid_domain_order():
    with pytest.raises(Exception):
        FuzzyVariable(domain=[1, 0])


def test_get_membership_value_out_of_bounds_index_negative():
    with pytest.raises(Exception):
        var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]), support_values=np.array([0.5]))
        var.get_membership_value_py(-2, 0)


def test_get_membership_value_out_of_bounds_index_big():
    with pytest.raises(Exception):
        var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]), support_values=np.array([0.5]))
        var.get_membership_value_py(1, 0)


def test_get_membership_value_empty_fuzzy_sets_array():
    with pytest.raises(Exception):
        var = FuzzyVariable()
        var.get_membership_value_py(0, 0)


def test_get_length_empty():
    var = FuzzyVariable()
    assert var.get_length() == 0


def test_get_fuzzy_set_value_out_of_bounds_index_negative():
    with pytest.raises(Exception):
        var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]), support_values=np.array([0.5]))
        var.get_fuzzy_set(-2)


def test_get_fuzzy_set_value_out_of_bounds_index_big():
    with pytest.raises(Exception):
        var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]), support_values=np.array([0.5]))
        var.get_fuzzy_set(1)


def test_get_fuzzy_set_value_empty_fuzzy_sets_array():
    with pytest.raises(Exception):
        var = FuzzyVariable()
        var.get_fuzzy_set(0)


def test_get_support_out_of_bounds_index_negative():
    with pytest.raises(Exception):
        var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]), support_values=np.array([0.5]))
        var.get_support(-2)


def test_get_support_out_of_bounds_index_big():
    with pytest.raises(Exception):
        var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]), support_values=np.array([0.5]))
        var.get_support(1)


def test_get_support_empty_fuzzy_sets_array():
    with pytest.raises(Exception):
        var = FuzzyVariable()
        var.get_support(0)


def test_eq_true():
    var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]),
                        support_values=np.array([0.5]), name="x0")
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]),
                        support_values=np.array([0.5]), name="x0")
    assert var == var2


def test_eq_different_fuzzy_sets_support_none():
    var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]),
                        support_values=np.array([0.5]), name="x0")
    var2 = FuzzyVariable(name="x0")
    assert var != var2


def test_eq_different_fuzzy_sets_support_size():
    var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small"),TriangularFuzzySet(0, 0.5, 1, 1, "small")]),
                        support_values=np.array([0.5, 0]), name="x0")
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]),
                        support_values=np.array([0.5]), name="x0")
    assert var != var2


def test_eq_different_fuzzy_sets_order():
    var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small"),
                                             TriangularFuzzySet(0, 0.5, 1, 1, "small")]),
                        support_values=np.array([0.5, 0]), name="x0")
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 1, "small"),
                                              TriangularFuzzySet(0, 0.5, 1, 0, "small")]),
                        support_values=np.array([0.5, 0]), name="x0")
    assert var != var2


def test_eq_different_support_values_order():
    var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small"),
                                             TriangularFuzzySet(0, 0.5, 1, 1, "small")]),
                        support_values=np.array([0.5, 0]), name="x0")
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small"),
                                              TriangularFuzzySet(0, 0.5, 1, 1, "small")]),
                        support_values=np.array([0, 0.5]), name="x0")
    assert var != var2


def test_eq_different_name():
    var = FuzzyVariable(name="x0")
    var2 = FuzzyVariable(name="x1")
    assert var != var2


def test_eq_different_domain():
    var = FuzzyVariable(domain=np.array([0.0, 1.0]))
    var2 = FuzzyVariable(domain=np.array([0.1, 0.5]))
    assert var != var2


def test_deepcopy():
    var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")]), support_values=np.array([0.5]), name="x0")
    var_copy = copy.deepcopy(var)

    assert (var == var_copy and
            id(var.get_support_values().base) != id(var_copy.get_support_values().base) and
            id(var.get_fuzzy_sets().base) != id(var_copy.get_fuzzy_sets().base))