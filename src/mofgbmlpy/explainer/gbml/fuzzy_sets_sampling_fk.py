from pymoo.core.sampling import Sampling
import numpy as np


class FuzzySetsSamplingFK(Sampling):
    def __init__(self, michigan_solution_builder):
        self._michigan_solution_builder = michigan_solution_builder

        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        solutions = self._michigan_solution_builder.create(num_solutions=n_samples)
        solutions = np.reshape(solutions, (n_samples, 1))
        return solutions
