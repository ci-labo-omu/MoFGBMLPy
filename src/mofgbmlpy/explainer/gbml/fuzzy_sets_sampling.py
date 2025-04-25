from pymoo.core.sampling import Sampling
import numpy as np


class FuzzySetsSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        initial_population = np.zeros((n_samples, problem.n_var), dtype=object)

        for i in range(n_samples):
            for j in range(0, problem.n_var, 3):
                # Triangular fuzzy set
                initial_population[i][j] = np.random.uniform(0, 1)
                initial_population[i][j + 1] = np.random.uniform(initial_population[i][j], 1)
                initial_population[i][j + 2] = np.random.uniform(initial_population[i][j+1], 1)

                j += 3

        return initial_population

