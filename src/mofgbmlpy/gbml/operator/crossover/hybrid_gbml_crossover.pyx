import copy

from pymoo.core.crossover import Crossover
import numpy as np

class HybridGBMLCrossover(Crossover):
    __min_num_rules = None
    __max_num_rules = None

    def __init__(self, random_gen, michigan_crossover_probability, michigan_crossover, pittsburgh_crossover, prob=0.9):
        super().__init__(2, 1, prob)
        self.__michigan_crossover_probability = michigan_crossover_probability
        self.__michigan_crossover = michigan_crossover
        self.__pittsburgh_crossover = pittsburgh_crossover
        self._random_gen = random_gen

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        michigan_crossover_mask = np.empty(n_matings, dtype=np.bool_)

        cdef bint at_least_one_michigan_crossover = False
        cdef bint at_least_one_pittsburgh_crossover = False

        for i in range(n_matings):
            # Check if the 2 parents are identical
            if X[0,i,0] == X[1,i,0]:
                michigan_crossover_mask[i] = True
                if not at_least_one_michigan_crossover:
                    at_least_one_michigan_crossover = True
            else:
                michigan_crossover_mask[i] = self._random_gen.random() < self.__michigan_crossover_probability
                if not at_least_one_michigan_crossover and michigan_crossover_mask[i]:
                    at_least_one_michigan_crossover = True
                elif not at_least_one_pittsburgh_crossover and not michigan_crossover_mask[i]:
                    at_least_one_pittsburgh_crossover = True

        if at_least_one_pittsburgh_crossover:
            Y_pittsburgh = self.__pittsburgh_crossover.execute(problem, X[:, np.invert(michigan_crossover_mask)], **kwargs)

        if at_least_one_michigan_crossover:
            Y_michigan = self.__michigan_crossover.execute(problem, X[0, michigan_crossover_mask], **kwargs)

        if at_least_one_michigan_crossover and at_least_one_pittsburgh_crossover:
            Y = np.concatenate((Y_michigan, Y_pittsburgh), axis=1)
        elif not at_least_one_michigan_crossover:
            return Y_pittsburgh
        elif not at_least_one_pittsburgh_crossover:
            return Y_michigan
        else:
            raise Exception("No offspring created during hybrid crossover")

        return Y
