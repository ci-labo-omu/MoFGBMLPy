import copy

import numpy as np

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.objectives.pittsburgh.error_rate import ErrorRate
from mofgbmlpy.gbml.objectives.pittsburgh.num_rules import NumRules
from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem
from mofgbmlpy.main.arguments import Arguments
from util import get_a0_0_iris_train_test

train, _ = get_a0_0_iris_train_test()


def test_deep_copy():
    # Just check if it raises an exception
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    consequent_factory = LearningBasic(train)

    obj = MichiganProblem(np.array([NumRules()]), 0, train, RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))
    _ = copy.deepcopy(obj)

    assert True
