from time import sleep

import numpy as np

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent import Consequent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic


def test_get_fuzzy_set_plot_data():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent = Antecedent(np.array([0, 3, 4]), knowledge)
    consequent = Consequent(ClassLabelBasic(1), RuleWeightBasic(0.3))
    rule = RuleBasic(antecedent, consequent)

    dim_index = 1

    mf = rule.get_fuzzy_set_object(dim_index).get_function()
    for i in mf.get_params():
        print(i, end=" ")
    print()
    print([mf.get_param_range(i) for i in range(len(mf.get_params()))])
    print(mf.get_plot_points())

    print(rule.get_fuzzy_set_object(dim_index).get_term())
    print(rule.get_var_concept(dim_index))
    # TODO
    assert True
