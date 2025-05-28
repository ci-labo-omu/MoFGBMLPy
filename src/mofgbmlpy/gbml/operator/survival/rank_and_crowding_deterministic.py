import numpy as np
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding


class RankAndCrowdingDeterministic(RankAndCrowding):
    def __init__(self):
        super().__init__()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            # always compute crowding distance for the front
            crowding = self.crowding_func.do(F[front, :])

            # assign rank and crowding to all individuals in the front
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding[j])

            if len(survivors) + len(front) > n_survive:
                needed = n_survive - len(survivors)

                # Deterministic descending sort by crowding distance
                sorted_indices = np.argsort(-crowding)[:needed]
                selected = [front[i] for i in sorted_indices]

                survivors.extend(selected)
                break  # we've filled the survivors list
            else:
                survivors.extend(front)

        return pop[survivors]
