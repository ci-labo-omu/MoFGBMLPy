import copy

import numpy as np
import pytest

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi import RuleWeightMulti


def test_none_rule_weight():
    with pytest.raises(TypeError):
        RuleWeightBasic(None)


def test_set_value_none_rule_weight():
    rw = RuleWeightBasic(0)
    with pytest.raises(TypeError):
        rw.set_value(None)


def test_set_value_rule_weight_invalid_type():
    rw = RuleWeightBasic(0)
    with pytest.raises(TypeError):
        rw.set_value(np.array([0.5, 0.1]))


def test_set_value_rule_weight_valid():
    rw = RuleWeightBasic(0)
    rw.set_value(0.1)

    assert rw.get_value() == 0.1


def test_eq_different_types():
    assert RuleWeightBasic(0.0) != RuleWeightMulti(np.array([0.0]))


def test_eq_different_content():
    assert RuleWeightBasic(0.0) != RuleWeightBasic(1.0)


def test_deep_copy():
    obj = RuleWeightBasic(0)
    copied_object = copy.deepcopy(obj)

    assert obj == copied_object
