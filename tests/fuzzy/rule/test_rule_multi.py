import copy

import numpy as np

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.class_label.class_label_multi import ClassLabelMulti
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent_multi import ConsequentMulti

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi import RuleWeightMulti
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic
from mofgbmlpy.fuzzy.rule.rule_multi import RuleMulti


def test_deep_copy():
    # Just check if it raises an exception
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent = Antecedent(np.array([0, 2, 1]), knowledge)
    consequent = ConsequentMulti(ClassLabelMulti(np.array([1, 0], int)), RuleWeightMulti(np.array([1.0, 1.0])))

    obj = RuleMulti(antecedent, consequent)
    _ = copy.deepcopy(obj)

    assert True