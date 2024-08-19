import xml.etree.cElementTree as xml_tree
import copy
from xml.dom import minidom

import numpy as np

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.input import Input
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_args import MoFGBMLNSGAIIArgs
from util import get_a0_0_iris_train_test, get_a0_0_german_train_test

training_data_set, _ = get_a0_0_iris_train_test()


training_data_set, _ = get_a0_0_iris_train_test()
training_data_set_multi, _ = get_a0_0_german_train_test()

#
#
# def test_get_length_none_list():
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = None
#     assert cl.get_length(solutions) == 0
#
#
# def test_get_length_empty_list():
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = np.empty(0, object)
#     assert cl.get_length(solutions) == 0
#
#
# def test_get_length_basic():
#     antecedent_indices = np.array([0 if i != 0 and i != 1 and i != 2 else 1 for i in range(training_data_set.get_num_dim())], int)
#     sol1 = create_michigan_sol(training_data_set, antecedent_indices=np.copy(antecedent_indices))
#
#     antecedent_indices = np.array([0 if i != 1 else 1 for i in range(training_data_set.get_num_dim())], int)
#     sol2 = create_michigan_sol(training_data_set, antecedent_indices=np.copy(antecedent_indices))
#
#     antecedent_indices = np.array([0 for _ in range(training_data_set.get_num_dim())], int)
#     sol3 = create_michigan_sol(training_data_set, antecedent_indices=np.copy(antecedent_indices))
#
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = np.array([sol1, sol2, sol3], object)
#
#     assert cl.get_length(solutions) == 4
#
#
# def test_get_length_multi():
#     is_attribute_categorical = [attr < 0 for attr in training_data_set_multi.get_pattern(0).get_attributes_vector()]
#
#     antecedent_indices = np.array([0 if i != 0 and i != 1 and i != 2 else (-1 if is_attribute_categorical[i] else 1) for i in range(training_data_set_multi.get_num_dim())], int)
#     sol1 = create_michigan_sol(training_data_set_multi, antecedent_indices=np.copy(antecedent_indices), is_multi_label=True)
#
#     antecedent_indices = np.array([0 if i != 1 else (-1 if is_attribute_categorical[i] else 1) for i in range(training_data_set_multi.get_num_dim())], int)
#     sol2 = create_michigan_sol(training_data_set_multi, antecedent_indices=np.copy(antecedent_indices), is_multi_label=True)
#
#     antecedent_indices = np.array([0 for _ in range(training_data_set_multi.get_num_dim())], int)
#     sol3 = create_michigan_sol(training_data_set_multi, antecedent_indices=np.copy(antecedent_indices), is_multi_label=True)
#
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = np.array([sol1, sol2, sol3], object)
#
#     assert cl.get_length(solutions) == 4
#
#
# def test_get_error_rate_none_list():
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = None
#
#     with pytest.raises(TypeError):
#         cl.get_error_rate_py(solutions, training_data_set)
#
#
# def test_get_error_rate_empty_list():
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = np.empty(0, object)
#
#     with pytest.raises(ValueError):
#         cl.get_error_rate_py(solutions, training_data_set)
#
#
# def test_get_error_rate_none_dataset():
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = np.empty(0, object)
#
#     with pytest.raises(TypeError):
#         cl.get_error_rate_py(solutions, None)
#
#
# class TestGetErrorRateBasic:
#     def test_get_error_rate_all_wins(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
#             Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelBasic(2)),
#             Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelBasic(2)),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelBasic(0)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelBasic(1)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         # for p in dataset.get_patterns():
#         #     print("fitness: ",sol2.get_fitness_value(p.get_attributes_vector()))
#         # print()
#
#         antecedent_indices = np.array([2, 2, 0, 0], int)
#         class_label = ClassLabelBasic(2)
#         rule_weight = RuleWeightBasic(0.5)
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         error_rate = cl.get_error_rate_py(solutions, dataset)
#
#         assert (error_rate == 0.0 and
#                 sol1.get_fitness() == 1 and sol1.get_num_wins() == 1 and
#                 sol2.get_fitness() == 1 and sol2.get_num_wins() == 1 and
#                 sol3.get_fitness() == 2 and sol3.get_num_wins() == 2)
#
#     def test_get_error_rate_all_but_one_wins(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
#             Pattern(2, np.array([1.0, 0.0, 0.5, 0.0]), ClassLabelBasic(2)),
#             Pattern(3, np.array([0.0, 1.0, 0.0, 0.5]), ClassLabelBasic(2)),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelBasic(0)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelBasic(1)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#
#         antecedent_indices = np.array([14, 0, 4, 0], int)
#         class_label = ClassLabelBasic(2)
#         rule_weight = RuleWeightBasic(0.5)
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         error_rate = cl.get_error_rate_py(solutions, dataset)
#         assert (error_rate == 1 / 4 and
#                 sol1.get_fitness() == 1 and sol1.get_num_wins() == 1 and
#                 sol2.get_fitness() == 1 and sol2.get_num_wins() == 1 and
#                 sol3.get_fitness() == 1 and sol3.get_num_wins() == 1)
#
#     def test_get_error_rate_all_wins_but_one_classification_error(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
#             Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelBasic(2)),
#             Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelBasic(0)),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelBasic(0)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelBasic(1)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#
#         antecedent_indices = np.array([2, 2, 0, 0], int)
#         class_label = ClassLabelBasic(2)
#         rule_weight = RuleWeightBasic(0.5)
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         error_rate = cl.get_error_rate_py(solutions, dataset)
#
#         assert (error_rate == 1 / 4 and
#                 sol1.get_fitness() == 1 and sol1.get_num_wins() == 1 and
#                 sol2.get_fitness() == 1 and sol2.get_num_wins() == 1 and
#                 sol3.get_fitness() == 1 and sol3.get_num_wins() == 2)
#
#
# class TestGetErrorRateMulti:
#     def test_get_error_rate_all_wins(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
#             Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
#             Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelMulti(np.array([0, 1, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#
#         antecedent_indices = np.array([2, 2, 0, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 1]))
#         rule_weight = RuleWeightMulti(np.array([0.5, 0.5, 0.5]))
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         error_rate = cl.get_error_rate_py(solutions, dataset)
#
#         assert (error_rate == 0.0 and
#                 sol1.get_fitness() == 1 and sol1.get_num_wins() == 1 and
#                 sol2.get_fitness() == 1 and sol2.get_num_wins() == 1 and
#                 sol3.get_fitness() == 2 and sol3.get_num_wins() == 2)
#
#     def test_get_error_rate_all_but_one_wins(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
#             Pattern(2, np.array([1.0, 0, 0.5, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
#             Pattern(3, np.array([0.0, 1.0, 0.0, 0.5]), ClassLabelMulti(np.array([1, 0, 1]))),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelMulti(np.array([0, 1, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([14, 0, 4, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 1]))
#         rule_weight = RuleWeightMulti(np.array([0.5, 0.5, 0.5]))
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         error_rate = cl.get_error_rate_py(solutions, dataset)
#         assert (error_rate == 1 / 4 and
#                 sol1.get_fitness() == 1 and sol1.get_num_wins() == 1 and
#                 sol2.get_fitness() == 1 and sol2.get_num_wins() == 1 and
#                 sol3.get_fitness() == 1 and sol3.get_num_wins() == 1)
#
#     def test_get_error_rate_all_wins_but_one_classification_error(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
#             Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
#             Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelMulti(np.array([0, 1, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([2, 2, 0, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 1]))
#         rule_weight = RuleWeightMulti(np.array([0.5, 0.5, 0.5]))
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         error_rate = cl.get_error_rate_py(solutions, dataset)
#
#         assert (error_rate == 1 / 4 and
#                 sol1.get_fitness() == 1 and sol1.get_num_wins() == 1 and
#                 sol2.get_fitness() == 1 and sol2.get_num_wins() == 1 and
#                 sol3.get_fitness() == 1 and sol3.get_num_wins() == 2)
#
#
# def test_get_errored_patterns_none_list():
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = None
#
#     with pytest.raises(TypeError):
#         cl.get_errored_patterns_py(solutions, training_data_set)
#
#
# def test_get_errored_patterns_empty_list():
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = np.empty(0, object)
#
#     with pytest.raises(ValueError):
#         cl.get_errored_patterns_py(solutions, training_data_set)
#
#
# def test_get_errored_patterns_none_dataset():
#     cl = Classifier(SingleWinnerRuleSelection())
#     solutions = np.empty(0, object)
#
#     with pytest.raises(TypeError):
#         cl.get_errored_patterns_py(solutions, None)
#
#
# class TestGetErroredPatternsBasic:
#     def test_get_errored_patterns_all_wins(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
#             Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelBasic(2)),
#             Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelBasic(2)),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelBasic(0)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelBasic(1)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         # for p in dataset.get_patterns():
#         #     print("fitness: ",sol2.get_fitness_value(p.get_attributes_vector()))
#         # print()
#
#         antecedent_indices = np.array([2, 2, 0, 0], int)
#         class_label = ClassLabelBasic(2)
#         rule_weight = RuleWeightBasic(0.5)
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         errored_patterns = cl.get_errored_patterns_py(solutions, dataset)
#
#         assert (len(errored_patterns) == 0)
#
#     def test_get_errored_patterns_all_but_one_wins(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
#             Pattern(2, np.array([1.0, 0.0, 0.5, 0.0]), ClassLabelBasic(2)),
#             Pattern(3, np.array([0.0, 1.0, 0.0, 0.5]), ClassLabelBasic(2)),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelBasic(0)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelBasic(1)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#
#         antecedent_indices = np.array([14, 0, 4, 0], int)
#         class_label = ClassLabelBasic(2)
#         rule_weight = RuleWeightBasic(0.5)
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         errored_patterns = cl.get_errored_patterns_py(solutions, dataset)
#
#         assert len(errored_patterns) == 1 and errored_patterns[0] == dataset.get_pattern(3)
#
#     def test_get_errored_patterns_all_wins_but_one_classification_error(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
#             Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelBasic(2)),
#             Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelBasic(0)),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelBasic(0)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelBasic(1)
#         rule_weight = RuleWeightBasic(0.7)
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#
#         antecedent_indices = np.array([2, 2, 0, 0], int)
#         class_label = ClassLabelBasic(2)
#         rule_weight = RuleWeightBasic(0.5)
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent))
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         errored_patterns = cl.get_errored_patterns_py(solutions, dataset)
#
#         assert len(errored_patterns) == 1 and errored_patterns[0] == dataset.get_pattern(3)
#
#
# class TestGetErroredPatternsMulti:
#     def test_get_errored_patterns_all_wins(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
#             Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
#             Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelMulti(np.array([0, 1, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#
#         antecedent_indices = np.array([2, 2, 0, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 1]))
#         rule_weight = RuleWeightMulti(np.array([0.5, 0.5, 0.5]))
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         errored_patterns = cl.get_errored_patterns_py(solutions, dataset)
#
#         assert len(errored_patterns) == 0
#
#     def test_get_errored_patterns_all_but_one_wins(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
#             Pattern(2, np.array([1.0, 0, 0.5, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
#             Pattern(3, np.array([0.0, 1.0, 0.0, 0.5]), ClassLabelMulti(np.array([1, 0, 1]))),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelMulti(np.array([0, 1, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([14, 0, 4, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 1]))
#         rule_weight = RuleWeightMulti(np.array([0.5, 0.5, 0.5]))
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         errored_patterns = cl.get_errored_patterns_py(solutions, dataset)
#         assert len(errored_patterns) == 1 and errored_patterns[0] == dataset.get_pattern(3)
#
#     def test_get_errored_patterns_all_wins_but_one_classification_error(self):
#         dataset = Dataset(4, 4, 3, np.array([
#             Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
#             Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
#             Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
#             Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
#         ]))
#
#         antecedent_indices = np.array([4, 0, 14, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol1 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([0, 4, 0, 14], int)
#         class_label = ClassLabelMulti(np.array([0, 1, 0]))
#         rule_weight = RuleWeightMulti(np.array([0.7, 0.7, 0.7]))
#         consequent = Consequent(class_label, rule_weight)
#         sol2 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         antecedent_indices = np.array([2, 2, 0, 0], int)
#         class_label = ClassLabelMulti(np.array([1, 0, 1]))
#         rule_weight = RuleWeightMulti(np.array([0.5, 0.5, 0.5]))
#         consequent = Consequent(class_label, rule_weight)
#         sol3 = create_michigan_sol(dataset, antecedent_indices=np.copy(antecedent_indices),
#                                    consequent=copy.deepcopy(consequent), is_multi_label=True)
#
#         solutions = np.array([sol1, sol2, sol3], object)
#
#         cl = Classifier(SingleWinnerRuleSelection())
#         errored_patterns = cl.get_errored_patterns_py(solutions, dataset)
#
#         assert len(errored_patterns) == 1 and errored_patterns[0] == dataset.get_pattern(3)

def test_deep_copy():
    # Just check if it raises an exception
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()

    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    consequent_factory = LearningBasic(training_data_set)
    michigan_solution_builder = MichiganSolutionBuilder(random_gen, 1, 0,
                                                        RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge))
    obj = PittsburghSolution(2, 2, 0, michigan_solution_builder, SingleWinnerRuleSelection())
    copied_obj = copy.deepcopy(obj)
    assert obj == copied_obj and id(obj.get_vars().base) != id(copied_obj.get_vars().base)

    for i in range(obj.get_num_vars()):
        v1 = obj.get_var(i)
        v2 = copied_obj.get_var(i)

        assert (v1 == v2 and id(v1) != id(v2) and
                id(v1.get_vars().base) != id(v2.get_vars().base) and
                id(v1.get_antecedent()) != id(v2.get_antecedent()) and
                id(v1.get_antecedent().get_antecedent_indices().base) != id(v2.get_antecedent().get_antecedent_indices().base))


def test_to_xml_run():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    # Only test if it doesn't return an exception
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()
    rule_builder = RuleBuilderBasic(AllCombinationAntecedentFactory(knowledge, random_gen), LearningBasic(training_data_set), knowledge)
    michigan_builder = MichiganSolutionBuilder(random_gen, 2, 0, rule_builder)
    classification = SingleWinnerRuleSelection()
    problem = PittsburghProblem(3, ["error-rate", "num-rules"], 0, training_data_set, michigan_builder, classification)
    sol = problem.create_solution()
    reparsed = minidom.parseString(xml_tree.tostring(sol.to_xml()))
    _ = reparsed.toprettyxml(indent="  ")

    assert True
