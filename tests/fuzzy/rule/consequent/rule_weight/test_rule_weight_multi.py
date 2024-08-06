import copy

import numpy as np

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi import RuleWeightMulti


def test_deep_copy():
    # Just check if it raises an exception
    obj = RuleWeightMulti(np.array([1.0, 2.0, 3.0]))
    _ = copy.deepcopy(obj)

    assert True