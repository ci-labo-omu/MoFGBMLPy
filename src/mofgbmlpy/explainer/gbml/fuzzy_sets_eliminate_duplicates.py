from pymoo.core.duplicate import DuplicateElimination
import numpy as np

from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem


class FuzzySetsEliminateDuplicates(DuplicateElimination):
    def __init__(self, problem, epsilon=1e-16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self._problem = problem

    @staticmethod
    def distance_mfs_params(params_1, params_2, threshold=1e-8):
        if len(params_1) != len(params_2):
            return max(len(params_1), len(params_2))

        distance = 0
        for fs_i in range(len(params_1)):
            if len(params_1[fs_i]) != len(params_2[fs_i]):
                distance += 1
            elif len(params_1[fs_i]) != 0:
                for i in range(len(params_1[fs_i])):
                    distance += 1 if abs(params_1[fs_i][i] - params_2[fs_i][i]) > threshold else 0

        return distance

    def calc_dist(self, pop):
        X = self.func(pop)

        distance = np.empty((len(X), len(X)))

        params_x = np.empty((len(X), X[0][0].get_rule().get_antecedent_array_size()), dtype=object)
        for i in range(len(X)):
            fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(X[i][0].get_rule())
            for j in range(len(fuzzy_sets)):
                params_x[i][j] = fuzzy_sets[j].get_function().get_params()

        for i in range(len(X)):
            for j in range(i, (len(X))):
                distance[i][j] = self.distance_mfs_params(params_x[i], params_x[j])
                if i != j:
                    distance[j][i] = distance[i][j]

        return distance

    def is_duplicate_in_other_pop(self, rule, other):
        rule_fs = CounterfactualProblem.get_fuzzy_sets_from_rule(rule)
        rule_params = np.empty(len(rule_fs), dtype=object)
        for k in range(len(rule_fs)):
            rule_params[k] = rule_fs[k].get_function().get_params()
        for j in range(len(other)):
            other_fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(other[j].X[0].get_rule())
            params_other = np.empty(len(other_fuzzy_sets), dtype=object)
            for k in range(len(other_fuzzy_sets)):
                params_other[k] = other_fuzzy_sets[k].get_function().get_params()

            if self.distance_mfs_params(params_other, rule_params) < self.epsilon:
                return True

    def _do(self, pop, other, is_duplicate):
        distance = self.calc_dist(pop)
        n = len(distance)
        is_duplicate = np.zeros(n, dtype=bool)

        for i in range(n):
            # distance between current solutions
            if np.any(distance[i, :i] < self.epsilon):
                is_duplicate[i] = True

            # distance to previous population
            if other is not None:
                is_duplicate[i] = self.is_duplicate_in_other_pop(pop[i].X[0].get_rule(), other)

        return is_duplicate
