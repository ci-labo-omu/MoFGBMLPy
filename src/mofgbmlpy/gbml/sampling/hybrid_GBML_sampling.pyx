import time

from pymoo.core.sampling import Sampling

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
import numpy as np
cimport numpy as cnp

from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution


# from cython.parallel cimport prange


class HybridGBMLSampling(Sampling):
    __learner = None

    def __init__(self, training_set):
        super().__init__()
        self.__learner = LearningBasic(training_set)

    def _do(self, problem, n_samples, **kwargs):
        cdef int i
        cdef AbstractSolution[:,:] initial_population = np.zeros((n_samples,1), dtype=object)

        for i in range(n_samples):
            initial_population[i][0] = problem.create_solution()

        return initial_population
