import time

from pymoo.core.sampling import Sampling

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
import numpy as np
cimport numpy as cnp
# from cython.parallel cimport prange


class HybridGBMLSampling(Sampling):
    __learner = None

    def __init__(self, training_set):
        super().__init__()
        self.__learner = LearningBasic(training_set)

    def _do(self, problem, n_samples, **kwargs):
        cdef cnp.ndarray[object, ndim=2] initial_population = np.zeros((n_samples,1), dtype=object)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     tasks = {executor.submit(problem.create_solution): _ for _ in range(n_samples)}
        #     i = 0
        #     for task in tasks:
        #         initial_population[i] = [task.result()]
        #         i += 1

        # times = np.zeros(n_samples)
        for i in range(n_samples):
            # start = time.time()
        # for i in prange(n_samples, nogil=True): # TODO CHECK IF IT CAN BE USED
            initial_population[i][0] = problem.create_solution()
            # print(initial_population[i][0].get_vars())
            # times[i] = time.time() - start
        # print(np.sum(times), times)


        return initial_population
