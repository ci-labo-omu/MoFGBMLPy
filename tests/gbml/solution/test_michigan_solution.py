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
from mofgbmlpy.utility.random import init_random_gen

init_random_gen(2020)


def test_hash():
    # Only test if it doesn't return an exception
    args = MoFGBMLNSGAIIArgs()

    args.set("TRAIN_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("TEST_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("IS_MULTI_LABEL", False)

    training_data_set, _ = Input.get_train_test_files(args)
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()

    antecedent_factory = AllCombinationAntecedentFactory(knowledge)
    consequent_factory = LearningBasic(training_data_set)
    sol1 = MichiganSolution(2, 0, RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))
    sol2 = MichiganSolution(2, 0, RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))
    sol3 = MichiganSolution(2, 0, RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))

    sol1.set_vars(np.array([0, 1, 2, 3], int))
    sol2.set_vars(np.array([0, 1, 2, 3], int))
    sol3.set_vars(np.array([0, 1, 3, 2], int))

    assert hash(sol1) == hash(sol2) and hash(sol1) != hash(sol3)
