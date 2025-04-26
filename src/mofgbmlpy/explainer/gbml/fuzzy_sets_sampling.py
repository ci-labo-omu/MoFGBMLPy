from pymoo.core.sampling import Sampling
import numpy as np
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
import copy


class FuzzySetsSampling(Sampling):
    def __init__(self, noise_str=0.1):
        self._noise_str = noise_str
        super().__init__()

    # def _do(self, problem, n_samples, **kwargs):
    #     initial_population = np.zeros((n_samples, problem.n_var), dtype=object)
    #
    #     for i in range(n_samples-1):
    #         for j in range(problem.n_var):
    #             # Triangular fuzzy set
    #
    #             left = np.random.uniform(0, 1)
    #             center = np.random.uniform(left, 1)
    #             right = np.random.uniform(center, 1)
    #
    #             initial_population[i, j] = TriangularFuzzySet(left, center, right, j, "new_term")
    #     initial_population[-1] = np.array([copy.deepcopy(fs) for fs in problem.get_initial_fuzzy_sets()])
    #     return initial_population

    def _do(self, problem, n_samples, **kwargs):
        initial_population = np.zeros((n_samples, problem.n_var), dtype=object)
        initial_fuzzy_sets = problem.get_initial_fuzzy_sets()
        initial_params = np.array([fs.get_function().get_params() for fs in initial_fuzzy_sets], dtype=object)

        for i in range(n_samples - 1):
            for j in range(problem.n_var):
                if isinstance(initial_fuzzy_sets[j], TriangularFuzzySet):
                    # Triangular fuzzy set
                    old_params = initial_params[j]

                    # add noise
                    left = old_params[0] + np.random.normal(0, self._noise_str)
                    center = old_params[1] + np.random.normal(0, self._noise_str)
                    right = old_params[2] + np.random.normal(0, self._noise_str)

                    # fix
                    left = max(0, min(left, 1))
                    center = max(left, min(center, 1))
                    right = max(center, min(right, 1))

                    initial_population[i, j] = TriangularFuzzySet(left, center, right, j, "new_term")
                else:
                    initial_population[i, j] = copy.deepcopy(problem.get_initial_fuzzy_sets()[j])
        initial_population[-1] = np.array([copy.deepcopy(fs) for fs in problem.get_initial_fuzzy_sets()])

        return initial_population
