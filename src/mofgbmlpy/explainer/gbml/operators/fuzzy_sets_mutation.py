import copy

from pymoo.core.mutation import Mutation
import numpy as np
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet


class FuzzySetsMutation(Mutation):
    def __init__(self, prob=1.0):
        super().__init__(prob=prob)

    def _do(self, problem, X, **kwargs):
        new_x = copy.deepcopy(X)

        for i in range(len(new_x)):
            for j in range(problem.n_var):
                # Triangular fuzzy set
                if isinstance(new_x[i][j], TriangularFuzzySet):
                    # We pick a random param to mutate
                    mutation_param = np.random.randint(0, 3)
                    mf = new_x[i][j].get_function()
                    params = mf.get_params()

                    previous_param = params[mutation_param - 1] if mutation_param > 0 else 0
                    next_param = params[mutation_param + 1] if mutation_param < 2 else 1

                    new_value = np.random.uniform(previous_param, next_param)

                    mf.set_param_value(mutation_param, new_value)

        return new_x
