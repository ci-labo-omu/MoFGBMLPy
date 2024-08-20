import copy

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.rule.consequent.consequent_basic import ConsequentBasic

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic


def test_none_class_label():
    class_label = None
    rule_weight = RuleWeightBasic(1.0)
    ConsequentBasic(class_label, rule_weight)


def test_none_rule_weight():
    class_label = ClassLabelBasic(0)
    rule_weight = None
    ConsequentBasic(class_label, rule_weight)


def test_deep_copy():
    # Just check if it raises an exception
    obj = ConsequentBasic(ClassLabelBasic(1), RuleWeightBasic(1))
    _ = copy.deepcopy(obj)

    assert True