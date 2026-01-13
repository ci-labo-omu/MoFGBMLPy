from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
import numpy as np
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
import copy

from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet

from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation


class FuzzySetsSamplingFK(Sampling):
    def __init__(self, knowledge, sampling_change_prob):
        random_gen = np.random.Generator(np.random.MT19937(seed=2022))
        self._mutation = MichiganMutation(knowledge, sampling_change_prob, random_gen)

        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        initial_population_X = np.zeros((n_samples, 1), dtype=object)

        initial_rule = copy.deepcopy(problem.get_factual_rule())
        initial_rule.resize_objectives(problem.n_obj)
        initial_rule.clear_attributes()
        initial_rule.reset_num_wins()
        initial_rule.reset_fitness()

        for i in range(n_samples):
            new_rule = copy.deepcopy(initial_rule)
            initial_population_X[i][0] = new_rule

        initial_population = Population.new("X", initial_population_X)
        initial_population = self._mutation.do(problem, initial_population)

        initial_population_X = initial_population.get("X")
        initial_population_X[0][0] = initial_rule

        return initial_population_X
