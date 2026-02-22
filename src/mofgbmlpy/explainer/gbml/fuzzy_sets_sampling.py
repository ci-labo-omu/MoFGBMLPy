from pymoo.core.sampling import Sampling
import numpy as np
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
import copy

from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet


class FuzzySetsSampling(Sampling):
    """Sampling method to create new solutions with fuzzy sets for the rules, by adding noise to the initial fuzzy sets and randomly changing their type.

    Attributes:
        _noise_str (float): The standard deviation of the noise added to the fuzzy sets parameters
        _change_fs_type_prob (float): The probability of changing the type of a fuzzy set to DontCareFuzzySet
        _change_fs_params_prob (float): The probability of changing the parameters of a fuzzy set by adding noise

    """

    def __init__(self, noise_str=0.1, change_fs_type_prob=0.0, change_fs_params_prob=1.0):
        """Constructor

        Args:
            noise_str (float, optional): The standard deviation of the noise added to the fuzzy sets parameters. Defaults to 0.1.
            change_fs_type_prob (float, optional): The probability of changing the type of a fuzzy set to DontCareFuzzySet. Defaults to 0.0.
            change_fs_params_prob (float, optional): The probability of changing the parameters of a fuzzy set by adding noise. Defaults to 1.0.
        """
        self._noise_str = noise_str
        self._change_fs_type_prob = change_fs_type_prob
        self._change_fs_params_prob = change_fs_params_prob
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        """Apply the sampling to create new solutions with fuzzy sets for the rules, by adding noise to the initial fuzzy sets and randomly changing their type.

        Args:
            problem (Problem): The optimization problem being solved, used to get the initial fuzzy sets if needed
            n_samples (int): The number of solutions to sample

        Returns:
            np.ndarray: An array of shape (n_samples, 1) containing the sampled solutions with fuzzy sets for the rules
        """
        initial_population = np.zeros((n_samples, 1), dtype=object)
        initial_rule = copy.deepcopy(problem.get_factual_rule())
        initial_rule.set_deep_copy_knowledge(True)  # since knowledge is not shared between individuals here
        initial_rule.resize_objectives(problem.n_obj)

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
                    if np.random.rand() < self._change_fs_type_prob:
                        fuzzy_sets[j] = DontCareFuzzySet(0)
                        continue

                    old_params = initial_params[j]

                    if np.random.rand() < self._change_fs_params_prob:
                        # add noise
                        left = old_params[0] + np.random.normal(0, self._noise_str)
                        center = old_params[1] + np.random.normal(0, self._noise_str)
                        right = old_params[2] + np.random.normal(0, self._noise_str)

                        # fix
                        left = max(0, min(left, 1))
                        center = max(left, min(center, 1))
                        right = max(center, min(right, 1))
                    else:
                        left, center, right = old_params

                    fuzzy_sets[j] = TriangularFuzzySet(left, center, right, 1, "new_term")
                    new_antecedent_indices[j] = 1
                elif isinstance(initial_fuzzy_sets[j], DontCareFuzzySet):
                    if np.random.rand() < self._change_fs_type_prob:
                        left = np.random.uniform(0, 0.5)
                        right = np.random.uniform(0.5, 1)
                        center = np.random.uniform(left, right)
                        fuzzy_sets[j] = TriangularFuzzySet(left, center, right, 1, "new_term")
                        new_antecedent_indices[j] = 1
                else:
                    raise NotImplementedError("Only TriangularFuzzySet and DontCareFuzzySet are supported.")

            initial_population[i][0].set_knowledge(CounterfactualProblem.build_knowledge(fuzzy_sets))
            initial_population[i][0].set_vars(new_antecedent_indices)
            initial_population[i][0].get_rule().get_antecedent().set_antecedent_indices(new_antecedent_indices)

        fuzzy_sets = np.empty(n_dim, dtype=object)
        for j in range(fuzzy_sets.shape[0]):
            fuzzy_sets[j] = copy.deepcopy(initial_fuzzy_sets[j])

        new_antecedent_indices = np.array(
            [1 if fs is not None and not isinstance(fs, DontCareFuzzySet) else 0 for fs in fuzzy_sets]
        )
        initial_population[0][0].set_knowledge(CounterfactualProblem.build_knowledge(fuzzy_sets))
        initial_population[0][0].set_vars(new_antecedent_indices)
        initial_population[0][0].get_rule().get_antecedent().set_antecedent_indices(new_antecedent_indices)

        return initial_population
