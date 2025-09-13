"""Pymoo problem class for the task offloading problem."""

import numpy as np
from pymoo.core.problem import Problem
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet


class CounterfactualProblem(Problem):
    def __init__(self, classifier, changed_rule_index, target_class):
        self._factual_michigan_solution = classifier.get_var(changed_rule_index)
        self._initial_class = self._factual_michigan_solution.get_class_label().get_class_label_value()
        self._target_class = target_class
        self._area_computation_num_samples = 100  # The higher it is, the more precise it gets, but it's also slower
        self._learner = self._factual_michigan_solution.get_rule_builder().get_consequent_factory()
        self._train_set = self._learner.get_training_set()

        self._initial_mfs_y = self.compute_membership_values(self._factual_michigan_solution, 0, 1)

        super().__init__(n_var=1, n_obj=2, n_eq_constr=1)

    def get_initial_mfs_y(self):
        return self._initial_mfs_y

    def get_factual_rule(self):
        return self._factual_michigan_solution

    def get_target_class(self):
        return self._target_class

    @staticmethod
    def get_fuzzy_sets_from_rule(rule):
        fs_list = [rule.get_fuzzy_set_object(dim) for dim in range(rule.get_antecedent_array_size())]
        return np.array(fs_list, dtype=object)

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
        # We want to minimize the confidence difference between the initial class
        # and the target class and we want to maximize the confidence of the target class

        # TODO: to be optimized, because for now all confidences are computed (add a function to compute only one confidence in the learner)

        antecedent = current_rule.get_antecedent()
        confidences = self._learner.calc_confidence_py(antecedent, self._train_set)

        confidence_target_class = confidences[self._target_class.get_class_label_value()]

        # max_conf = np.max(confidences)

        # confidence_loss = 1/(1 + np.exp(-(max_conf-confidence_target_class**2-confidence_target_class)))
        confidence_loss = 1 - confidence_target_class

        return confidence_loss

    def change_loss(self, current_rule):
        # Change loss
        change_loss = 0

        iou = self.compute_iou_with_factual(current_rule)

        if iou is not None:
            change_loss = 1 - np.mean(iou)

        return change_loss

    def is_output_class_target(self, current_rule):
        return current_rule.get_class_label() == self._target_class

    def objectives(self, current_rule):
        confidence_loss = self.conf_loss(current_rule)
        change_loss = self.change_loss(current_rule)

        # print(f"conf loss: {confidence_loss}, change_loss: {change_loss}")
        return confidence_loss, change_loss

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.empty((len(X), 2))
        out["H"] = np.empty((len(X),))

        for i, ind in enumerate(X):
            ind[0].learning()
            rule = ind[0]
            out["F"][i][0] = self.conf_loss(rule)
            out["F"][i][1] = self.change_loss(rule)
            rule.set_objective(0, out["F"][i][0])
            rule.set_objective(1, out["F"][i][1])
            out["H"][i] = 0 if self.is_output_class_target(rule) else 1  # constraint

    @staticmethod
    def get_objective_names():
        return ["1 - target class confidence", "1 - IoU"]
