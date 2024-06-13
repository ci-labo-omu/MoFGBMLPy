import copy

from pymoo.core.crossover import Crossover
import numpy as np
import random

import time


class HybridGBMLCrossover(Crossover):
    __min_num_rules = None
    __max_num_rules = None

    def __init__(self, michigan_crossover_probability, michigan_crossover, pittsburgh_crossover, prob=0.9):
        super().__init__(2, 1, prob)
        self.__michigan_crossover_probability = michigan_crossover_probability
        self.__michigan_crossover = michigan_crossover
        self.__pittsburgh_crossover = pittsburgh_crossover

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        start = time.time()

        michigan_crossover_mask = np.empty(n_matings, dtype=np.bool_)

        for i in range(n_matings):
            # Check if the 2 parents are identical
            if X[0,i,0] == X[1,i,0]:
                michigan_crossover_mask[i] = True
            else:
                michigan_crossover_mask[i] = random.random() < self.__michigan_crossover_probability

        Y_michigan = self.__michigan_crossover.execute(problem, X[0, michigan_crossover_mask], **kwargs)
        Y_pittsburgh = self.__pittsburgh_crossover.execute(problem, X[:, michigan_crossover_mask], **kwargs)
        Y = np.concatenate((Y_michigan, Y_pittsburgh), axis=1)

        elapsed = time.time() - start  # 11.96s
        return Y
