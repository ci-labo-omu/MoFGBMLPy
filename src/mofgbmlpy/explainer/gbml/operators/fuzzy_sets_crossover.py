import copy

from pymoo.core.crossover import Crossover
import numpy as np

class FuzzySetsCrossover(Crossover):
    def __init__(self, prob=0.5):
        super().__init__(2, 2, prob=prob)

    def _do(self, problem, X, **kwargs):
        print(X.shape)
        raise NotImplementedError("Crossover not implemented yet")
        _, n_matings, _ = X.shape
        offsprings = np.zeros((1, n_matings, 1), dtype=object)

        for i in range(n_matings):
            p1 = X[0, i, 0]
            p2 = X[1, i, 0]

