import copy

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.rule.consequent.consequent import Consequent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic


def test_deep_copy():
    # Just check if it raises an exception
    obj = Consequent(ClassLabelBasic(1), RuleWeightBasic(1))
    _ = copy.deepcopy(obj)

    assert True