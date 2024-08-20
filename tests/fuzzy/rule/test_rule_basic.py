import copy

import numpy as np

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent_basic import ConsequentBasic

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic


def test_deep_copy():
    # Just check if it raises an exception
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent = Antecedent(np.array([0, 2, 1]), knowledge)
    consequent = ConsequentBasic(ClassLabelBasic(1), RuleWeightBasic(1))

    obj = RuleBasic(antecedent, consequent)
    _ = copy.deepcopy(obj)

    assert True