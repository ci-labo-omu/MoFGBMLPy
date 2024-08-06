import copy

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic


def test_deep_copy():
    # Just check if it raises an exception
    obj = RuleWeightBasic(0)
    _ = copy.deepcopy(obj)

    assert True