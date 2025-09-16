"""
The code in this file is mainly copied from the Pymoo library,
since the function _do of the RankAndCrowding class is protected
"""

import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function


class FuzzySetsSurvival(Survival):
    def __init__(self, eliminate_duplicates=None, nds=None, crowding_func="cd"):
        self._eliminate_duplicates = eliminate_duplicates
        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            indices_I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(indices_I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=n_remove)

                indices_I = randomized_argsort(crowding_of_front, order="descending", method="numpy")
                indices_I = indices_I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=0)

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[indices_I])

        survivors = pop[survivors]
        if self._eliminate_duplicates is not None:
            survivors = self._eliminate_duplicates.do(survivors)

        return survivors
