from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
import numpy as np
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
import copy

from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet

from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation


class FuzzySetsSamplingFK(Sampling):
    def __init__(self, michigan_solution_builder):
        self._michigan_solution_builder = michigan_solution_builder

        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        solutions = self._michigan_solution_builder.create(num_solutions=n_samples)
        solutions = np.reshape(solutions, (n_samples, 1))
        return solutions
