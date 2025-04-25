"""Pymoo problem class for the task offloading problem."""

import numpy as np
from pymoo.core.problem import Problem


class CounterfactualProblem(Problem):
    def __init__(self, knowledge, fuzzy_rule, initial_class, target_class, learner):
        self._knowledge = knowledge
        self._fuzzy_rule = fuzzy_rule
        self._initial_class = initial_class
        self._target_class = target_class
        self._area_computation_num_samples = 100
        self._learner = learner
        self._train_set = learner.get_training_set()

        antecedent = fuzzy_rule.get_antecedent()
        antecedent_indices = antecedent.get_antecedent_indices()

        fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
        for i, idx in enumerate(antecedent_indices):
            fuzzy_sets[i] = self._knowledge.get_fuzzy_set(i, idx)

        self._initial_params = [fs.get_function().get_params() for fs in fuzzy_sets]

        n_vars = 0

        # TODO: not optimized or simple enough but use it for now
        # We map var indices to the dimension it corresponds to (used to get the corresponding fuzzy set and mf)
        self._vars_to_dim = {}
        self._fuzzy_set_size = {}
        for i, fuzzy_set in enumerate(fuzzy_sets):
            num_params = len(self._initial_params[i])

            offset = i * num_params
            for j in range(num_params):
                self._vars_to_dim[offset + j] = i

            self._fuzzy_set_size[i] = num_params

            n_vars += num_params

        super().__init__(
            n_var=n_vars, n_obj=2, xl=0, xu=1
        )

        self._initial_membership_values = self.compute_membership_values(fuzzy_sets, 0, 1)

    def var_to_dim(self, var_index):
        return self._vars_to_dim[var_index]

    def get_fuzzy_set_size(self, dim_index):
        return self._fuzzy_set_size[dim_index]

    def get_fuzzy_set(self, fuzzy_set_index):
        # Get the fuzzy set corresponding to the given index
        fuzzy_sets = self._fuzzy_rule.get_antecedent().get_fuzzy_sets()
        return fuzzy_sets[fuzzy_set_index]

    def compute_membership_values(self, fuzzy_sets, min_val=0, max_val=1):
        mfs = [fs.get_function() for fs in fuzzy_sets]

        x_samples = np.linspace(min_val, max_val, self._area_computation_num_samples)
        mf_values = np.zeros((len(fuzzy_sets), len(x_samples)), dtype=object)
        for i, mf in enumerate(mfs):
            mf_values[i] = [mf.get_value_py(x) for x in x_samples]

        return mf_values

    def compute_membership_area_data(self, mf_1_y, mf_2_y, step):
        union_value = np.zeros(mf_1_y.shape[0])
        intersection_value = np.zeros(mf_1_y.shape[0])
        mf_1_highest_area = np.zeros(mf_1_y.shape[0])
        mf_2_highest_area = np.zeros(mf_1_y.shape[0])

        for fs_i in range(mf_1_y.shape[0]):
            intersection_value[fs_i] = 0
            union_value[fs_i] = 0
            mf_1_highest_area[fs_i] = 0
            mf_2_highest_area[fs_i] = 0

            for i in range(mf_1_y.shape[1]):
                y_val = max(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                union_value[fs_i] += step * y_val

                y_val = min(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                intersection_value[fs_i] += step * y_val

                if mf_1_y[fs_i][i] > mf_2_y[fs_i][i]:
                    mf_1_highest_area[fs_i] += step
                else:
                    mf_2_highest_area[fs_i] += step

        return intersection_value, union_value, mf_1_highest_area, mf_2_highest_area

    def objectives(self, x, fuzzy_sets):
        # x are membership functions params here

        # Confidence loss
        # We want to minimize the confidence difference between the initial class and the target class and we want to maximize the confidence of the target class

        # TODO: to be optimized, because for now all confidences are computed
        confidences = self._learner.calc_confidence_py(self._fuzzy_rule.get_antecedent(), self._train_set)

        confidence_initial_class = confidences[self._initial_class.get_class_label_value()]
        confidence_target_class = confidences[self._target_class.get_class_label_value()]

        diff_loss_part = 2 / (1 + np.exp(confidence_initial_class - confidence_target_class) ** 2)
        y_value_loss_part = np.exp(-2 * confidence_target_class)
        confidence_loss = diff_loss_part + y_value_loss_part

        # Change loss
        change_loss = 0

        current_mf_values = self.compute_membership_values(fuzzy_sets, 0, 1)
        step = 1 / current_mf_values.shape[1]

        intersection_values, union_values, mf_1_highest_areas, mf_2_highest_areas = (
            self.compute_membership_area_data(self._initial_params, x, step)
        )

        if intersection_values is not None and union_values is not None:
            for i in range(len(intersection_values)):
                change_loss += 1 - (intersection_values[i] / union_values[i])
                # print(f"Intersection: {intersection_values[i]}, Union: {union_values[i]}, Change Loss: {change_loss}")

            change_loss /= len(intersection_values)

        return confidence_loss, change_loss

    def _evaluate(self, x, out, fuzzy_sets):
        out["F"] = [self.objectives(ind, fuzzy_sets) for ind in x]
