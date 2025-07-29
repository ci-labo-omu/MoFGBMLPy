"""Pymoo problem class for the task offloading problem."""

import numpy as np
from pymoo.core.problem import Problem
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet


class CounterfactualProblem(Problem):
    def __init__(self, knowledge, fuzzy_rule, initial_class, target_class, learner):
        self._fuzzy_rule = fuzzy_rule
        self._initial_class = initial_class
        self._target_class = target_class
        self._area_computation_num_samples = 100
        self._learner = learner
        self._train_set = learner.get_training_set()

        antecedent = fuzzy_rule.get_antecedent()
        antecedent_indices = antecedent.get_antecedent_indices()

        n_vars = len(antecedent_indices)

        self._initial_fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
        for i, idx in enumerate(antecedent_indices):
            self._initial_fuzzy_sets[i] = knowledge.get_fuzzy_set(i, idx)

        self._initial_mfs_y = self.compute_membership_values(self._initial_fuzzy_sets, 0, 1)

        super().__init__(n_var=n_vars, n_obj=2, xl=0, xu=1, n_eq_constr=1)

    def get_initial_mfs_y(self):
        return self._initial_mfs_y

    def get_fuzzy_rule(self):
        return self._fuzzy_rule

    def get_target_class(self):
        return self._target_class

    def get_initial_fuzzy_sets(self):
        return self._initial_fuzzy_sets

    def compute_membership_values(self, fuzzy_sets, min_val=0, max_val=1):
        mfs = [fs.get_function() for fs in fuzzy_sets]

        x_samples = np.linspace(min_val, max_val, self._area_computation_num_samples)
        mf_values = np.zeros((len(fuzzy_sets), len(x_samples)), dtype=object)
        for i, mf in enumerate(mfs):
            mf_values[i] = [mf.get_value_py(x) for x in x_samples]

        return mf_values

    def compute_iou(self, mf_1_y, mf_2_y, step):
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
    def build_antecedent(fuzzy_sets):
        antecedent_indices = np.ones(len(fuzzy_sets), dtype=int)
        fuzzy_vars = np.empty(len(fuzzy_sets), dtype=object)

        for i in range(len(fuzzy_sets)):
            if len(fuzzy_sets[i].get_function().get_params()) == 0:
                # DC
                antecedent_indices[i] = 0
                fuzzy_vars[i] = FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0)]), name=f"x{i}")
            else:
                fuzzy_vars[i] = FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0), fuzzy_sets[i]]), name=f"x{i}")
        knowledge = Knowledge(fuzzy_vars)

        antecedent = Antecedent(antecedent_indices, knowledge)

        return antecedent

    def objectives(self, fuzzy_sets):
        antecedent = self.build_antecedent(fuzzy_sets)

        # Confidence loss
        # We want to minimize the confidence difference between the initial class
        # and the target class and we want to maximize the confidence of the target class

        # TODO: to be optimized, because for now all confidences are computed
        confidences = self._learner.calc_confidence_py(antecedent, self._train_set)

        confidence_target_class = confidences[self._target_class.get_class_label_value()]

        # max_conf = np.max(confidences)

        # confidence_loss = 1/(1 + np.exp(-(max_conf-confidence_target_class**2-confidence_target_class)))
        confidence_loss = 1 - confidence_target_class

        # Change loss
        change_loss = 0

        current_mf_values = self.compute_membership_values(fuzzy_sets, 0, 1)
        step = 1 / current_mf_values.shape[1]

        iou = self.compute_iou(self._initial_mfs_y, current_mf_values, step)

        if iou is not None:
            change_loss = 1 - np.mean(iou)

        output_class_is_target = np.argmax(confidences) == self._target_class.get_class_label_value()

        # print(f"conf loss: {confidence_loss}, change_loss: {change_loss}")
        return confidence_loss, change_loss, output_class_is_target

    def build_rule(self, fuzzy_sets):
        antecedent = self.build_antecedent(fuzzy_sets)
        consequent = self._learner.learning(antecedent)
        rule = RuleBasic(antecedent, consequent)

        # rule.plot_antecedent()

        return rule

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.empty((len(X), 2))
        out["H"] = np.empty((len(X),))

        for i, ind in enumerate(X):
            conf_loss, change_loss, output_class_is_target = self.objectives(ind)
            out["F"][i][0] = conf_loss
            out["F"][i][1] = change_loss
            out["H"][i] = 0 if output_class_is_target else 1  # constraint

    def get_objective_names(self):
        return ["1 - target class confidence", "1 - IoU"]
