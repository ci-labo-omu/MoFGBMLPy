import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask


class UniformCrossoverSingleOffspring(Crossover):

    def __init__(self, **kwargs, prob):
        super().__init__(2, 1, prob=prob, **kwargs)

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < 0.5
        _X = crossover_mask(X, M)

        # Select one offspring randomly
        selected = np.random.choice(_X, size=1)

        return selected