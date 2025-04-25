from pymoo.core.mutation import Mutation
import numpy as np


class FuzzySetsMutation(Mutation):
    def __init__(self, prob=1.0):
        super().__init__(prob=prob)

    def _do(self, problem, X, **kwargs):
        new_x = X.copy()
        for i in range(len(X)):
            j = 0
            fuzzy_set_index = 0

            while j < problem.n_var:
                fuzzy_set_index_size = problem.fuzzy_set_index(fuzzy_set_index)
                if fuzzy_set_index_size == 3:  # triangular
                    # We pick a random param to mutate
                    mutation_param = np.random.randint(0, 3)
                    previous_param = new_x[i][j-1] if mutation_param > 0 else 0
                    next_param = new_x[i][j+1] if mutation_param < 2 else mutation_param

                    new_x[i][j] = np.random.uniform(previous_param, next_param)

                    j += 2
                    fuzzy_set_index += 1

                j += 1

        return new_x
