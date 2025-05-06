import copy

from pymoo.core.mutation import Mutation
import numpy as np
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.dont_care_mf import DontCareMF


class FuzzySetsMutation(Mutation):
    def __init__(self, prob=1.0, prob_mutated_param=0.2, prob_change_type=0.1):
        super().__init__(prob=prob)
        self._prob_mutated_param = prob_mutated_param
        self._prob_change_type = prob_change_type

    def _do(self, problem, X, **kwargs):
        new_x = copy.deepcopy(X)

        # new_type_idx = None

        for i in range(len(new_x)):
            for j in range(problem.n_var):
                if np.random.rand() < self._prob_change_type:
                    # new_type_idx = i
                    if isinstance(new_x[i][j].get_function(), TriangularMF):
                        # To DC
                        new_x[i][j] = DontCareFuzzySet(0)
                    elif isinstance(new_x[i][j].get_function(), DontCareMF):
                        # To triangular
                        left = np.random.rand()
                        center = np.random.rand() * (1 - left) + left
                        right = np.random.rand() * (1 - center) + center

                        new_x[i][j] = TriangularFuzzySet(left, center, right, 1, "new_term")
                else:
                    # Triangular fuzzy set
                    if isinstance(new_x[i][j], TriangularFuzzySet):
                        for param_i in range(3):
                            # We pick a random param to mutate
                            if np.random.rand() < self._prob_mutated_param:
                                mf = new_x[i][j].get_function()
                                params = mf.get_params()

                                previous_param = params[param_i - 1] if param_i > 0 else 0
                                next_param = params[param_i + 1] if param_i < 2 else 1

                                new_value = np.random.uniform(previous_param, next_param)

                                mf.set_param_value(param_i, new_value)

        # if new_type_idx is not None:
        #     problem.build_antecedent(new_x[new_type_idx]).plot_antecedent()

        return new_x
