from pymoo.core.duplicate import DuplicateElimination
import numpy as np


class FuzzySetsEliminateDuplicates(DuplicateElimination):
    def __init__(self, problem, epsilon=1e-16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self._problem = problem

    def calc_dist(self, pop):
        # TODO: use instead membership params to compute distance

        X = self.func(pop)

        x_mf_values = np.array([self._problem.compute_membership_values(x, 0, 1) for x in X])

        n = len(x_mf_values)
        distance = np.empty((n, n))

        step = 1 / x_mf_values.shape[1]

        for i in range(n):
            i_values = x_mf_values[i]
            for j in range(i, n):
                j_values = x_mf_values[j]
                iou = self._problem.compute_iou(i_values, j_values, step)
                dist_ij = 1.0 - np.mean(iou)
                distance[i, j] = dist_ij
                distance[j, i] = dist_ij

        return distance

    def _do(self, pop, other, is_duplicate):
        distance = self.calc_dist(pop)
        n = len(distance)
        is_duplicate = np.zeros(n, dtype=bool)

        for i in range(n):
            if np.any(distance[i, :i] < self.epsilon):
                is_duplicate[i] = True

        return is_duplicate
