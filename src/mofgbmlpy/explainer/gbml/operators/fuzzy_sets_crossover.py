import copy

from pymoo.core.crossover import Crossover
import numpy as np


class FuzzySetsCrossover(Crossover):
    def __init__(self, prob=0.5, p1_prob_off_1=0.5):
        super().__init__(2, 2, prob=prob)
        self._p1_prob_off_1 = p1_prob_off_1

    def _do(self, problem, X, **kwargs):
        _, n_matings, num_vars = X.shape
        offsprings = np.zeros((2, n_matings, num_vars), dtype=object)

        for i in range(n_matings):
            p1 = X[0, i, :]
            p2 = X[1, i, :]

            # p1 and p2 are lists of fuzzy sets
            # here we swap them (between dims)

            for j in range(num_vars):
                if np.random.rand() < self._p1_prob_off_1:
                    offsprings[0, i, j] = copy.deepcopy(p1[j])
                    offsprings[1, i, j] = copy.deepcopy(p2[j])
                else:
                    offsprings[0, i, j] = copy.deepcopy(p2[j])
                    offsprings[1, i, j] = copy.deepcopy(p1[j])

        return offsprings
