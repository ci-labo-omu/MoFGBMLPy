from pymoo.core.sampling import Sampling

from fuzzy.rule.consequent.learning.learning_basic import LearningBasic
import numpy as np
import time
import concurrent.futures


class HybridGBMLSampling(Sampling):
    __learner = None

    def __init__(self, training_set):
        super().__init__()
        self.__learner = LearningBasic(training_set)

    def _do(self, problem, n_samples, **kwargs):
        initial_population = np.zeros(n_samples, dtype=object)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     tasks = {executor.submit(problem.create_solution): _ for _ in range(n_samples)}
        #     i = 0
        #     for task in tasks:
        #         initial_population[i] = [task.result()]
        #         i += 1

        for i in range(n_samples):
            initial_population[i] = [problem.create_solution()]

        return initial_population
