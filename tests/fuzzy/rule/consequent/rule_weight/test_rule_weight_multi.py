import copy

import numpy as np
import pytest

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi import RuleWeightMulti


def test_none_rule_weight():
    with pytest.raises(TypeError):
        RuleWeightMulti(None)


def test_empty_rule_weight():
    with pytest.raises(ValueError):
        RuleWeightMulti(np.empty(0))


def test_get_rule_weight_at_out_of_bounds_negative():
    rw = RuleWeightMulti(np.array([0.5, 0.8]))
    with pytest.raises(IndexError):
        rw.get_rule_weight_at(-3)


def test_get_rule_weight_at_out_of_bounds_big():
    rw = RuleWeightMulti(np.array([0.5, 0.8]))
    with pytest.raises(IndexError):
        rw.get_rule_weight_at(2)


def test_get_rule_weight_at_valid():
    rw = RuleWeightMulti(np.array([0.5, 0.8]))
    assert rw.get_rule_weight_at(0) == 0.5
    assert rw.get_rule_weight_at(1) == 0.8


def test_set_value_none_rule_weight():
    rw = RuleWeightMulti(np.array([0.5, 0.8]))
    with pytest.raises(TypeError):
        rw.set_value(None)


def test_set_value_empty_rule_weight():
    rw = RuleWeightMulti(np.array([0.5, 0.8]))
    with pytest.raises(ValueError):
        rw.set_value(np.empty(0))


def test_set_value_rule_weight_invalid_type():
    rw = RuleWeightMulti(np.array([0.5, 0.8]))
    with pytest.raises(TypeError):
        rw.set_value(0)


def test_set_value_rule_weight_valid():
    rw = RuleWeightMulti(np.array([0.5, 0.8]))
    rw.set_value(np.array([0.1, 0.2]))

    val = rw.get_value()

    assert len(val) == 2
    assert val[0] == 0.1
    assert val[1] == 0.2


def test_get_mean_one():
    rw = RuleWeightMulti(np.array([0.8]))
    assert rw.get_mean_py() == 0.8


def test_get_mean():
    values = np.array([0.8, 0.5, 0.4, 0.5, 0.1])
    rw = RuleWeightMulti(values)
    assert rw.get_mean_py() == np.mean(values)


def test_eq_different_types():
    assert RuleWeightBasic(0.0) != RuleWeightMulti(np.array([0.0]))


def test_eq_different_content_order():
    assert RuleWeightMulti(np.array([1.0, 0.0])) != RuleWeightMulti(np.array([0.0, 1.0]))


def test_eq_different_content():
    assert RuleWeightMulti(np.array([1.0, 0.0])) != RuleWeightMulti(np.array([1.0, 2.0]))


def test_deep_copy():
    obj = RuleWeightMulti(np.array([1.0, 0.0]))
    copied_object = copy.deepcopy(obj)

    assert obj == copied_object and id(obj.get_value().base) != id(copied_object.get_value().base)