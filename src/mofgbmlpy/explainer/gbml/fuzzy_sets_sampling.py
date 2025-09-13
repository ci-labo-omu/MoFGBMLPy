from pymoo.core.sampling import Sampling
import numpy as np
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
import copy

from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet


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
        initial_population = np.zeros((n_samples, 1), dtype=object)
        initial_rule = problem.get_factual_rule()
        initial_rule.set_deep_copy_knowledge(True)  # since knowledge is not shared between individuals here

        initial_population[0] = [copy.deepcopy(initial_rule)]

        initial_fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(initial_rule.get_rule())
        initial_params = np.array([fs.get_function().get_params() for fs in initial_fuzzy_sets], dtype=object)

        n_dim = initial_rule.get_rule().get_antecedent_array_size()

        for i in range(1, n_samples):
            initial_population[i] = [copy.deepcopy(initial_rule)]

            fuzzy_sets = np.empty(n_dim, dtype=object)
            new_antecedent_indices = np.zeros(n_dim, dtype=int)

            for j in range(new_antecedent_indices.shape[0]):
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

                    fuzzy_sets[j] = TriangularFuzzySet(left, center, right, 1, "new_term")
                    new_antecedent_indices[j] = 1
                elif isinstance(initial_fuzzy_sets[j], DontCareFuzzySet):
                    continue
                else:
                    raise NotImplementedError("Only TriangularFuzzySet and DontCareFuzzySet are supported.")

            initial_population[i][0].set_knowledge(CounterfactualProblem.build_knowledge(fuzzy_sets))
            initial_population[i][0].set_vars(new_antecedent_indices)
            initial_population[i][0].get_rule().get_antecedent().set_antecedent_indices(new_antecedent_indices)

        fuzzy_sets = np.empty(n_dim, dtype=object)
        for j in range(fuzzy_sets.shape[0]):
            fuzzy_sets[j] = copy.deepcopy(initial_fuzzy_sets[j])

        new_antecedent_indices = np.array([1 if fs is not None and not isinstance(fs, DontCareFuzzySet) else 0 for fs in fuzzy_sets])
        initial_population[0][0].set_knowledge(CounterfactualProblem.build_knowledge(fuzzy_sets))
        initial_population[0][0].set_vars(new_antecedent_indices)
        initial_population[0][0].get_rule().get_antecedent().set_antecedent_indices(new_antecedent_indices)

        return initial_population
