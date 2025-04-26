from pymoo.core.duplicate import DuplicateElimination
import numpy as np


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
                    distance += 1 if abs(params_1[fs_i][i]-params_2[fs_i][i]) > threshold else 0

        return distance

    def calc_dist(self, pop):
        X = self.func(pop)

        distance = np.empty((len(X), len(X)))

        params_x = np.empty((len(X), len(X[0])), dtype=object)
        for i in range(len(X)):
            ind = X[i]
            for j in range(len(ind)):
                params_x[i][j] = ind[j].get_function().get_params()

        for i in range(len(X)):
            for j in range(i, (len(X))):
                distance[i][j] = self.distance_mfs_params(params_x[i], params_x[j])
                if i != j:
                    distance[j][i] = distance[i][j]

        return distance

    def _do(self, pop, other, is_duplicate):
        distance = self.calc_dist(pop)
        n = len(distance)
        is_duplicate = np.zeros(n, dtype=bool)

        for i in range(n):
            if np.any(distance[i, :i] < self.epsilon):
                is_duplicate[i] = True

        return is_duplicate
