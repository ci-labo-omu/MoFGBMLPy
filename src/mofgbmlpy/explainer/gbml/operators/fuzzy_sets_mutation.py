import copy

from pymoo.core.mutation import Mutation
import numpy as np
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.dont_care_mf import DontCareMF

from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem


class FuzzySetsMutation(Mutation):
    def __init__(self, prob=1.0, prob_mutated_param=0.2, prob_change_type=0.1, prob_revert_to_initial=0.0):
        super().__init__(prob=prob)
        self._prob_mutated_param = prob_mutated_param
        self._prob_change_type = prob_change_type
        self._prob_revert_to_initial = prob_revert_to_initial

    def _do(self, problem, X, **kwargs):
        new_x = copy.deepcopy(X)

        if self._prob_revert_to_initial > 0:
            initial_fuzzy_sets = problem.get_initial_fuzzy_sets()

        for i in range(len(new_x)):
            rule = new_x[i][0]
            fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(rule.get_rule())
            new_antecedent_indices = np.copy(rule.get_rule().get_antecedent().get_antecedent_indices())

            for j in range(len(fuzzy_sets)):
                if np.random.rand() < self._prob_revert_to_initial:
                    fuzzy_sets[j] = copy.deepcopy(initial_fuzzy_sets[j])
                    if isinstance(fuzzy_sets[j], TriangularFuzzySet):
                        new_antecedent_indices[j] = 1
                    elif isinstance(fuzzy_sets[j], DontCareFuzzySet):
                        new_antecedent_indices[j] = 0

                    continue

                if np.random.rand() < self._prob_change_type:
                    if isinstance(fuzzy_sets[j].get_function(), TriangularMF):
                        # To DC
                        fuzzy_sets[j] = DontCareFuzzySet(0)
                        new_antecedent_indices[j] = 0
                    elif isinstance(fuzzy_sets[j].get_function(), DontCareMF):
                        # To triangular
                        left = np.random.rand()
                        center = np.random.rand() * (1 - left) + left
                        right = np.random.rand() * (1 - center) + center

                        fuzzy_sets[j] = TriangularFuzzySet(left, center, right, 1, "new_term")
                        new_antecedent_indices[j] = 1
                else:
                    # Triangular fuzzy set
                    if isinstance(fuzzy_sets[j], TriangularFuzzySet):
                        for param_i in range(3):
                            # We pick a random param to mutate
                            if np.random.rand() < self._prob_mutated_param:
                                mf = fuzzy_sets[j].get_function()
                                params = mf.get_params()

                                previous_param = params[param_i - 1] if param_i > 0 else 0
                                next_param = params[param_i + 1] if param_i < 2 else 1

                                new_value = np.random.uniform(previous_param, next_param)

                                mf.set_param_value(param_i, new_value)

            new_x[i][0].set_knowledge(CounterfactualProblem.build_knowledge(fuzzy_sets))
            new_x[i][0].set_vars(new_antecedent_indices)
            new_x[i][0].get_rule().get_antecedent().set_antecedent_indices(new_antecedent_indices)
        return new_x
