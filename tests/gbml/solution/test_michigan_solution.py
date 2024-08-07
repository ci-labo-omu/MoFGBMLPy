import copy

import numpy as np

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.input import Input
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.classifier.classifier import Classifier
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_args import MoFGBMLNSGAIIArgs
from util import get_a0_0_iris_train_test

training_data_set, _ = get_a0_0_iris_train_test()

def test_hash():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()

    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    consequent_factory = LearningBasic(training_data_set)
    sol1 = MichiganSolution(random_gen, 2, 0, RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))
    sol2 = MichiganSolution(random_gen, 2, 0, RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))
    sol3 = MichiganSolution(random_gen, 2, 0, RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))

    sol1.set_vars(np.array([0, 1, 2, 3], int))
    sol2.set_vars(np.array([0, 1, 2, 3], int))
    sol3.set_vars(np.array([0, 1, 3, 2], int))

    assert hash(sol1) == hash(sol2) and hash(sol1) != hash(sol3)

def test_deep_copy():
    # Just check if it raises an exception
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()

    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    consequent_factory = LearningBasic(training_data_set)
    obj = MichiganSolution(random_gen, 2, 0, RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))
    _ = copy.deepcopy(obj)
