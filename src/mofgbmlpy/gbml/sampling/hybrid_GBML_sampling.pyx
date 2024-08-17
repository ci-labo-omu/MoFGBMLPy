import time

from pymoo.core.sampling import Sampling

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
import numpy as np
cimport numpy as cnp

from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution


# from cython.parallel cimport prange


class HybridGBMLSampling(Sampling):
    """Generate an initial population of Pittsburgh solutions

    Attributes:
        __learner(AbstractLearning): Learner used to generate rules
    """
    def __init__(self, learner):
        """Constructor

        Args:
            learner(AbstractLearning): Learner used to generate rules
        """
        super().__init__()
        self.__learner = learner

    def _do(self, problem, n_samples, **kwargs):
        """Run the population initialization operator

        Args:
            problem (Problem): Object used to define the optimization problem (used by Pymoo)
            n_samples (int): number of solutions to be generated
            **kwargs (dict): Other arguments used by Pymoo

        Returns:
            AbstractSolution[,]: Population of generated solutions as an array of shape (n_samples, 1)
        """
        cdef int i
        cdef AbstractSolution[:,:] initial_population = np.zeros((n_samples,1), dtype=object)

        for i in range(n_samples):
            initial_population[i][0] = problem.create_solution()

        return initial_population
