import copy
import random

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask


class UniformCrossoverSingleOffspring(Crossover):

    def __init__(self, prob, **kwargs):
        super().__init__(2, 1, prob=prob, **kwargs)

    def _do(self, _, X, **kwargs):
        n_parents, n_matings, _ = X.shape

        if n_parents != 2:
            raise Exception("Error: 2 parents are needed for this crossover")

        n_vars = X[0, 0, 0].get_num_vars()
        mask = np.random.random((n_matings, n_vars)) < 0.5

        new_shape = (1,) + X.shape[1:]
        offspring = np.empty(new_shape, dtype=X.dtype)

        for i in range(n_matings):
            child_1 = copy.deepcopy(X[0, i, 0])
            child_2 = copy.deepcopy(X[1, i, 0])
            # TODO clear objectives ?
            indices_1 = child_1.get_vars()
            indices_2 = child_2.get_vars()

            # print("\nOFFSPRING 1 (start): ", end="")
            # for xi in child_1.get_vars():
            #     print(xi, end=" ")
            #
            # print("\nOFFSPRING 2 (start): ", end="")
            # for xi in child_2.get_vars():
            #     print(xi, end=" ")
            #
            # print("\nPARENT 1: ", end="")
            # for xi in X[0, i, 0].get_vars():
            #     print(xi, end=" ")
            #
            # print("\nPARENT 2: ", end="")
            # for xi in X[1, i, 0].get_vars():
            #     print(xi, end=" ")


            for j in range(n_vars):
                if mask[i, j]:
                    tmp = indices_1[j]
                    indices_1[j] = indices_2[j]
                    indices_2[j] = tmp

            # Select one offspring randomly
            if random.random() < 0.5:
                offspring[0, i, 0] = child_1
            else:
                offspring[0, i, 0] = child_2

            indices = offspring[0, i, 0].get_antecedent().get_antecedent_indices()

            # print("\nPARENT 1 (end): ", end="")
            # for xi in X[0, i, 0].get_vars():
            #     print(xi, end=" ")
            #
            # print("\nPARENT 2 (end): ", end="")
            # for xi in X[1, i, 0].get_vars():
            #     print(xi, end=" ")
            #
            # print("\nOFFSPRING 1 (end): ", end="")
            # for xi in child_1.get_vars():
            #     print(xi, end=" ")
            #
            # print("\nOFFSPRING 2 (end): ", end="")
            # for xi in child_2.get_vars():
            #     print(xi, end=" ")
            #
            # print("\nOFFSPRING (end): ", end="")
            # for xi in offspring[0, i, 0].get_vars():
            #     print(xi, end=" ")
            # print()


        return offspring