"""Pymoo problem class for the task offloading problem."""

import copy

import numpy as np
from pymoo.core.problem import Problem
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet

from mofgbmlpy.explainer.util import append_rule_classifier

class CounterfactualProblem(Problem):
    def __init__(
        self, classifier, changed_rule_index, target_class, test_set, objectives=["confidence_loss", "change_loss"]
    ):
        self._classifier_copy_mutable = copy.deepcopy(classifier)
        self._classifier_copy = copy.deepcopy(classifier)
        self._changed_rule_index = changed_rule_index
        self._factual_michigan_solution = classifier.get_var(changed_rule_index)
        self._initial_class = self._factual_michigan_solution.get_class_label().get_class_label_value()
        self._target_class = target_class
        self._area_computation_num_samples = 50  # The higher it is, the more precise it gets, but it's also slower
        self._learner = self._factual_michigan_solution.get_rule_builder().get_consequent_factory()
        self._train_set = self._learner.get_training_set()
        self._test_set = test_set

        self._initial_mfs_y = self.compute_membership_values(self._factual_michigan_solution, 0, 1)

        self._objectives_map = {
            "confidence_loss": self.conf_loss,
            "change_loss": self.change_loss,
            "num_changed_features": self.num_changed_features_loss,
            "train_error_rate": self.error_rate,
            "rule_length": lambda rule: rule.get_length(),
        }

        self._objectives_map = {k: v for k, v in self._objectives_map.items() if k in objectives}

        if len(self._objectives_map) == 0:
            raise ValueError("At least one objective must be selected")

        super().__init__(n_var=1, n_obj=len(self._objectives_map), n_eq_constr=1)

    def get_factual_rule(self):
        return self._factual_michigan_solution

    def get_initial_num_rules(self):
        return self._classifier_copy.get_num_vars()

    def get_target_class(self):
        return self._target_class

    @staticmethod
    def get_fuzzy_sets_from_rule(rule):
        fs_list = [rule.get_fuzzy_set_object(dim) for dim in range(rule.get_antecedent_array_size())]
        return np.array(fs_list, dtype=object)

    def get_initial_fuzzy_sets(self):
        return CounterfactualProblem.get_fuzzy_sets_from_rule(self._factual_michigan_solution.get_rule())

    def compute_membership_values(self, rule, min_val=0, max_val=1):
        fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(rule.get_rule())
        mfs = [fs.get_function() for fs in fuzzy_sets]

        x_samples = np.linspace(min_val, max_val, self._area_computation_num_samples)
        mf_values = np.zeros((len(fuzzy_sets), len(x_samples)), dtype=object)
        for i, mf in enumerate(mfs):
            mf_values[i] = [mf.get_value_py(x) for x in x_samples]

        return mf_values

    def compute_iou_with_factual(self, rule):
        step = 1 / self._area_computation_num_samples

        mf_1_y = self._initial_mfs_y
        mf_2_y = self.compute_membership_values(rule)

        union_value = np.zeros(mf_1_y.shape[0])
        intersection_value = np.zeros(mf_1_y.shape[0])

        for fs_i in range(mf_1_y.shape[0]):
            intersection_value[fs_i] = 0
            union_value[fs_i] = 0

            for i in range(mf_1_y.shape[1]):
                y_val = max(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                union_value[fs_i] += step * y_val

                y_val = min(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                intersection_value[fs_i] += step * y_val

            if union_value[fs_i] == 0:
                union_value[fs_i] = 1
                intersection_value[fs_i] = 1

        return intersection_value / union_value

    @staticmethod
    def build_knowledge(fuzzy_sets):
        antecedent_indices = np.ones(len(fuzzy_sets), dtype=int)
        fuzzy_vars = np.empty(len(fuzzy_sets), dtype=object)

        for i in range(len(fuzzy_sets)):
            if fuzzy_sets[i] is None or len(fuzzy_sets[i].get_function().get_params()) == 0:
                # DC
                antecedent_indices[i] = 0
                fuzzy_vars[i] = FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0)]), name=f"x{i}")
            else:
                fuzzy_vars[i] = FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0), fuzzy_sets[i]]), name=f"x{i}")
        knowledge = Knowledge(fuzzy_vars)

        return knowledge

    def conf_loss(self, current_rule):
        # Confidence loss

        antecedent = current_rule.get_antecedent()
        confidences = self._learner.calc_confidence_py(antecedent, self._train_set)

        confidence_target_class = confidences[self._target_class.get_class_label_value()]

        confidence_loss = 1 - confidence_target_class

        return confidence_loss

    def change_loss(self, current_rule):
        # Change loss
        change_loss = 0

        iou = self.compute_iou_with_factual(current_rule)

        if iou is not None:
            change_loss = 1 - np.mean(iou)

        return change_loss

    def num_changed_features_loss(self, current_michigan_solution):
        num_changed_features = 0
        current_rule = current_michigan_solution.get_rule()
        factual_rule = self._factual_michigan_solution.get_rule()

        for i in range(current_rule.get_antecedent_array_size()):
            fs1 = factual_rule.get_fuzzy_set_object(i)
            fs2 = current_rule.get_fuzzy_set_object(i)

            if CounterfactualProblem.are_fuzzy_set_different(fs1, fs2):
                num_changed_features += 1

        return num_changed_features

    def _create_new_classifier(self, sol, replace=True):
        new_classifier = self._classifier_copy_mutable if replace else copy.deepcopy(self._classifier_copy)

        if replace:
            new_classifier.set_var(self._changed_rule_index, sol)
        else:
            new_classifier = append_rule_classifier(new_classifier, sol, train_set=self._train_set, deepcopy=False)
        return new_classifier

    def error_rate(self, current_rule=None, use_test_set=False, replace=True, initial_classifier=False):
        dataset = self._test_set if use_test_set else self._train_set

        if initial_classifier:
            return self._classifier_copy.calc_error_rate(self._train_set)

        if current_rule.get_rule().is_rejected_class_label():
            return 1.0

        new_classifier = self._create_new_classifier(current_rule, replace=replace)

        return new_classifier.calc_error_rate(dataset)

    def is_output_class_target(self, current_rule):
        return current_rule.get_class_label() == self._target_class

    def get_objectives(self, current_rule):
        objectives = np.empty(self.n_obj)
        for i, func in enumerate(self._objectives_map.values()):
            objectives[i] = func(current_rule)

        return objectives

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.empty((len(X), self.n_obj))
        out["H"] = np.empty((len(X),))

        for i, ind in enumerate(X):
            ind[0].learning()
            rule = ind[0]
            for j in range(self.n_obj):
                out["F"][i] = self.get_objectives(rule)
                rule.set_objective(j, out["F"][i][j])
            out["H"][i] = (
                0 if self.is_output_class_target(rule) else 1
            )  # constraint, note that rejected class labels are also removed here

    def get_objective_names(self):
        return list(self._objectives_map.keys())

    def num_wins_and_successes(self, sol=None, replace=False, initial_classifier=False):
        # success is defined as fitness in the code of MoFGBML

        if initial_classifier:
            new_classifier = self._classifier_copy
        else:
            new_classifier = self._create_new_classifier(sol, replace=replace)
            new_classifier.update_winners_and_errors(self._train_set)

        cl_vars = new_classifier.get_vars()
        num_wins = np.empty(len(cl_vars), dtype=int)
        fitness_vals = np.empty(len(cl_vars), dtype=int)

        for i, rule in enumerate(cl_vars):
            num_wins[i] = rule.get_num_wins()
            fitness_vals[i] = rule.get_fitness()

        return num_wins, fitness_vals

    @staticmethod
    def are_fuzzy_set_different(fs1, fs2, threshold=1e-8):
        params_1 = fs1.get_function().get_params()
        params_2 = fs2.get_function().get_params()

        if params_1 is None:
            params_1 = []
        if params_2 is None:
            params_2 = []

        if len(params_1) != len(params_2):
            return True

        for i in range(len(params_1)):
            if abs(params_1[i] - params_2[i]) > threshold:
                return True

        return False

    def get_changed_rule_index(self):
        return self._changed_rule_index

    def get_train_set(self):
        return self._train_set
