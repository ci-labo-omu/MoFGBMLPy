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


def get_training_set():
    args = Arguments()
    args.set("TRAIN_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("TEST_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("IS_MULTI_LABEL", False)
    train, _ = Input.get_train_test_files(args)
    return train


def test_deep_copy():
    # Just check if it raises an exception
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    consequent_factory = LearningBasic(get_training_set())

    obj = MichiganProblem(1, np.array([NumRules()]), 0, get_training_set(), RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))
    _ = copy.deepcopy(obj)

    assert True
