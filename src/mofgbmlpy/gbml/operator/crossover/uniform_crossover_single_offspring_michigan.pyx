import copy

import numpy as np
from mofgbmlpy.gbml.operator.crossover.pymoo_deepcopy_crossover import PymooDeepcopyCrossover
from pymoo.util.misc import crossover_mask



class UniformCrossoverSingleOffspringMichigan(PymooDeepcopyCrossover):
    """Uniform crossover for Michigan solutions"""

    def __init__(self, random_gen, prob, **kwargs):
        """Constructor

        Args:
            random_gen (numpy.random.Generator): Random generator
            prob (float): Crossover probability
            **kwargs (dict): Other Pymoo arguments
        """
        super().__init__(2, 1, random_gen, prob=prob, **kwargs)

    def _do(self, _, X, **kwargs):
        """Run the crossover on the given population

        Args:
            problem (Problem): Optimization problem (e.g. MichiganProblem)
            X (object[,]): Population. The shape is (n_parents, n_matings, n_var),
            **kwargs (dict): Other arguments taken by Pymoo crossover object

        Returns:
            double[,,]: Crossover offspring. Shape: (1, n_matings, n_vars)
        """
        n_parents, n_matings, _ = X.shape

        if n_parents != 2:
            raise ValueError("Error: 2 parents are needed for this crossover")

        n_vars = X[0, 0, 0].get_num_vars()
        mask = self._random_gen.random((n_matings, n_vars)) < 0.5

        new_shape = (1,) + X.shape[1:]
        offspring = np.empty(new_shape, dtype=X.dtype)

        for i in range(n_matings):
            child_1 = copy.deepcopy(X[0, i, 0])
            child_2 = copy.deepcopy(X[1, i, 0])

            indices_1 = child_1.get_vars()
            indices_2 = child_2.get_vars()

            for j in range(n_vars):
                if mask[i, j]:
                    tmp = indices_1[j]
                    indices_1[j] = indices_2[j]
                    indices_2[j] = tmp

            # Select one offspring randomly
            if self._random_gen.random() < 0.5:
                offspring[0, i, 0] = child_1
            else:
                offspring[0, i, 0] = child_2

            offspring[0, i, 0].reset_fitness()
            offspring[0, i, 0].reset_num_wins()

        return offspring