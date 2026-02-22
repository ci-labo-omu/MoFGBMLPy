"""The code in this file is mainly copied from the Pymoo library, since the function _do of the RankAndCrowding class is protected"""

import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function

from mofgbmlpy.explainer.gbml.crowding_function_x import CrowdingFunctionX


class FuzzySetsSurvival(Survival):
    """Survival operator that performs non-dominated sorting and crowding distance selection

    Attributes:
        eliminate_duplicates: Optional function to eliminate duplicates from the survivors
        nds: Non-dominated sorting method to use
        use_search_space_crowding (bool): Whether to use crowding in the search space (X) in addition to the objective space (F)
        crowding_func_x (CrowdingFunctionX): Crowding function for the search space, used if use_search_space_crowding is True
    """

    def __init__(self, eliminate_duplicates=None, nds=None, use_search_space_crowding=False):
        """Constructor

        Args:
            eliminate_duplicates: Optional function to eliminate duplicates from the survivors
            nds: Non-dominated sorting method to use (default is NonDominatedSorting)
            use_search_space_crowding (bool): Whether to use crowding in the search space (X) in addition to the objective space (F)
        """
        super().__init__(filter_infeasible=True)

        self._eliminate_duplicates = eliminate_duplicates
        self.use_search_space_crowding = use_search_space_crowding
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = get_crowding_function("cd")
        if self.use_search_space_crowding:
            self.crowding_func_x = CrowdingFunctionX()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        """Select best individual using non-dominated sorting and crowding distance.

        Args:
            problem: The optimization problem being solved.
            pop: The population of solutions to select from.
            n_survive: The number of individuals to survive.

        Returns:
            Population: The selected population of survivors.
        """
        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)
        if self.use_search_space_crowding:
            X = pop.get("X")

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
                if self.use_search_space_crowding:
                    crowding_of_front += self.crowding_func_x.do(X[front, :], n_remove=n_remove)

                indices_I = randomized_argsort(crowding_of_front, order="descending", method="numpy")
                indices_I = indices_I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=0)
                if self.use_search_space_crowding:
                    crowding_of_front += self.crowding_func_x.do(X[front, :], n_remove=0)

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
