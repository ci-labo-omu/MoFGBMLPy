import copy

import numpy as np
import pytest

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.class_label.class_label_multi import ClassLabelMulti
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.consequent import Consequent
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi import RuleWeightMulti
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from util import get_a0_0_iris_train_test, create_michigan_sol, get_a0_0_german_train_test


class TestSingleWinnerRuleSelectionBasic:
    training_data_set, _ = get_a0_0_iris_train_test()

    def test_classify_none_solutions_list(self):
        classification = SingleWinnerRuleSelection()
        solutions = None
        pattern = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_empty_solutions_list(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.empty(0, object)
        pattern = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_solutions_list_none_items(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([None], object)
        pattern = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelBasic(0))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_rejected_solutions_list(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set)], object)
        solutions[0].get_rule().get_consequent().set_rejected()
        pattern = Pattern(0, np.array([float(i) for i in range(self.training_data_set.get_num_dim())]),
                          ClassLabelBasic(0))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_none_pattern(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set)], object)
        pattern = None

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_empty_pattern_attributes(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set)], object)
        pattern = Pattern(0, np.empty(0), ClassLabelBasic(0))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_incompatible_dimensions_1(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set)], object)
        pattern = Pattern(0, np.array([float(i) for i in range(self.training_data_set.get_num_dim() + 1)]),
                          ClassLabelBasic(0))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_incompatible_dimensions_2(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set)], object)
        pattern = Pattern(0, np.array([float(i) for i in range(self.training_data_set.get_num_dim() - 1)]),
                          ClassLabelBasic(0))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_one_winner(self):
        classification = SingleWinnerRuleSelection()
        sol = create_michigan_sol(self.training_data_set)

        solutions = np.array([sol], object)
        pattern = self.training_data_set.get_pattern(0)

        winner = classification.classify(solutions, pattern)

        assert winner is not None and id(sol) == id(winner)

    def test_classify_same_fitness_values_different_class(self):
        antecedent_indices = np.array([0 if i != 1 else 1 for i in range(self.training_data_set.get_num_dim())],
                                      int)
        class_label = ClassLabelBasic(0)
        rule_weight = RuleWeightBasic(0.7)
        consequent = Consequent(class_label, rule_weight)
        sol1 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices), consequent=copy.deepcopy(consequent))

        class_label = ClassLabelBasic(1)
        rule_weight = RuleWeightBasic(0.7)
        consequent = Consequent(class_label, rule_weight)
        sol2 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices), consequent=copy.deepcopy(consequent))

        class_label = ClassLabelBasic(3)
        rule_weight = RuleWeightBasic(0.5)
        consequent = Consequent(class_label, rule_weight)
        sol3 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices), consequent=copy.deepcopy(consequent))

        solutions = np.array([sol1, sol2, sol3], object)
        pattern = Pattern(0, np.array([i / 10 for i in range(self.training_data_set.get_num_dim())]),
                          ClassLabelBasic(0))

        classification = SingleWinnerRuleSelection()
        assert classification.classify(solutions, pattern) is None

    def test_classify_same_fitness_values_same_class(self):
        antecedent_indices = np.array([0 if i != 1 else 1 for i in range(self.training_data_set.get_num_dim())], int)
        class_label = ClassLabelBasic(0)
        rule_weight = RuleWeightBasic(0.7)
        consequent = Consequent(class_label, rule_weight)
        sol1 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices), consequent=copy.deepcopy(consequent))

        class_label = ClassLabelBasic(0)
        rule_weight = RuleWeightBasic(0.7)
        consequent = Consequent(class_label, rule_weight)
        sol2 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices), consequent=copy.deepcopy(consequent))

        class_label = ClassLabelBasic(1)
        rule_weight = RuleWeightBasic(0.5)
        consequent = Consequent(class_label, rule_weight)
        sol3 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices), consequent=copy.deepcopy(consequent))

        solutions = np.array([sol1, sol2, sol3], object)
        pattern = Pattern(0, np.array([i / 10 for i in range(self.training_data_set.get_num_dim())]),
                          ClassLabelBasic(0))

        classification = SingleWinnerRuleSelection()
        assert id(classification.classify(solutions, pattern)) == id(sol1)  # 1st maximum fitness value

    def test_deep_copy(self):
        obj = SingleWinnerRuleSelection()
        obj_copy = copy.deepcopy(obj)

        assert obj == obj_copy


class TestSingleWinnerRuleSelectionMulti:
    training_data_set, _ = get_a0_0_german_train_test()

    def test_classify_none_solutions_list(self):
        classification = SingleWinnerRuleSelection()
        solutions = None
        pattern = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelMulti(np.array([0, 1])))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_empty_solutions_list(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.empty(0, object)
        pattern = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelMulti(np.array([0, 1])))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_solutions_list_none_items(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([None], object)
        pattern = Pattern(0, np.array([0.0, 1.0, 2.0]), ClassLabelMulti(np.array([0, 1])))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_rejected_solutions_list(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set, is_multi_label=True)], object)
        solutions[0].get_rule().get_consequent().set_rejected()
        pattern = Pattern(0, np.array([float(i) for i in range(self.training_data_set.get_num_dim())]),
                          ClassLabelMulti(np.array([0, 1])))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_none_pattern(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set, is_multi_label=True)], object)
        pattern = None

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_empty_pattern_attributes(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set, is_multi_label=True)], object)
        pattern = Pattern(0, np.empty(0), ClassLabelMulti(np.array([0, 1])))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_incompatible_dimensions_1(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set, is_multi_label=True)], object)
        pattern = Pattern(0, np.array([float(i) for i in range(self.training_data_set.get_num_dim() + 1)]),
                          ClassLabelMulti(np.array([0, 1])))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_incompatible_dimensions_2(self):
        classification = SingleWinnerRuleSelection()
        solutions = np.array([create_michigan_sol(self.training_data_set, is_multi_label=True)], object)
        pattern = Pattern(0, np.array([float(i) for i in range(self.training_data_set.get_num_dim() - 1)]),
                          ClassLabelMulti(np.array([0, 1])))

        with pytest.raises(Exception):
            classification.classify(solutions, pattern)

    def test_classify_one_winner(self):
        classification = SingleWinnerRuleSelection()
        sol = create_michigan_sol(self.training_data_set, is_multi_label=True)

        solutions = np.array([sol], object)
        pattern = self.training_data_set.get_pattern(0)

        winner = classification.classify(solutions, pattern)

        assert winner is not None and id(sol) == id(winner)

    def test_classify_same_fitness_values_different_class(self):
        antecedent_indices = np.array([0 if i != 1 else 1 for i in range(self.training_data_set.get_num_dim())], int)
        class_label = ClassLabelMulti(np.array([0, 1]))
        rule_weight = RuleWeightMulti(np.array([0.7, 0.7]))
        consequent = Consequent(class_label, rule_weight)
        sol1 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices),
                                   consequent=copy.deepcopy(consequent), is_multi_label=True)

        class_label = ClassLabelMulti(np.array([1, 1]))
        rule_weight = RuleWeightMulti(np.array([0.7, 0.7]))
        consequent = Consequent(class_label, rule_weight)
        sol2 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices),
                                   consequent=copy.deepcopy(consequent), is_multi_label=True)

        class_label = ClassLabelMulti(np.array([0, 0]))
        rule_weight = RuleWeightMulti(np.array([0.5, 0.5]))
        consequent = Consequent(class_label, rule_weight)
        sol3 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices),
                                   consequent=copy.deepcopy(consequent), is_multi_label=True)

        solutions = np.array([sol1, sol2, sol3], object)
        pattern = Pattern(0, np.array([i / 10 for i in range(self.training_data_set.get_num_dim())]),
                          ClassLabelMulti(np.array([0, 1])))

        classification = SingleWinnerRuleSelection()
        assert classification.classify(solutions, pattern) is None

    def test_classify_same_fitness_values_same_class(self):
        antecedent_indices = np.array([0 if i != 1 else 1 for i in range(self.training_data_set.get_num_dim())], int)
        class_label = ClassLabelMulti(np.array([0, 1]))
        rule_weight = RuleWeightMulti(np.array([0.7, 0.7]))
        consequent = Consequent(class_label, rule_weight)
        sol1 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices),
                                   consequent=copy.deepcopy(consequent), is_multi_label=True)

        class_label = ClassLabelMulti(np.array([0, 1]))
        rule_weight = RuleWeightMulti(np.array([0.7, 0.7]))
        consequent = Consequent(class_label, rule_weight)
        sol2 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices),
                                   consequent=copy.deepcopy(consequent), is_multi_label=True)

        class_label = ClassLabelMulti(np.array([1, 1]))
        rule_weight = RuleWeightMulti(np.array([0.5, 0.5]))
        consequent = Consequent(class_label, rule_weight)
        sol3 = create_michigan_sol(self.training_data_set, antecedent_indices=np.copy(antecedent_indices),
                                   consequent=copy.deepcopy(consequent), is_multi_label=True)

        solutions = np.array([sol1, sol2, sol3], object)
        pattern = Pattern(0, np.array([i / 10 for i in range(self.training_data_set.get_num_dim())]),
                          ClassLabelMulti(np.array([0, 1])))

        classification = SingleWinnerRuleSelection()
        assert id(classification.classify(solutions, pattern)) == id(sol1)  # 1st maximum fitness value

    def test_deep_copy(self):
        obj = SingleWinnerRuleSelection()
        obj_copy = copy.deepcopy(obj)

        assert obj == obj_copy
