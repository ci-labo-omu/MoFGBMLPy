from pymoo.core.sampling import Sampling
import numpy as np


class FuzzySetsSamplingFK(Sampling):
    """Sampling method to create new solutions with fuzzy sets for the rules, using a Michigan solution builder.

    Attributes:
        michigan_solution_builder (MichiganSolutionBuilder): Builder to create Michigan solutions with fuzzy sets for the sampling
    """

    def __init__(self, michigan_solution_builder):
        """Constructor

        Args:
            michigan_solution_builder (MichiganSolutionBuilder): Builder to create Michigan solutions with fuzzy sets for the sampling
        """
        self._michigan_solution_builder = michigan_solution_builder

        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        """Apply the sampling to create new solutions with fuzzy sets for the rules.

        Args:
            problem (Problem): The optimization problem being solved, used to get the initial fuzzy sets if needed
            n_samples (int): The number of solutions to sample
        """
        solutions = self._michigan_solution_builder.create(num_solutions=n_samples)
        solutions = np.reshape(solutions, (n_samples, 1))
        return solutions
