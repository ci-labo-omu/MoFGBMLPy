import numpy as np
from pymoo.operators.selection.tournament import TournamentSelection


class NaryTournamentSelectionOnFitness(TournamentSelection):
    @staticmethod
    def nary_fitness_tournament(pop, P, **kwargs):
        n_tournaments, n_candidates = P.shape

        if n_candidates < 0:
            raise ValueError("tournament_size must be positive")
        elif n_candidates == 1:
            return P # TODO: test this conditional branch

        S = np.full(n_tournaments, np.nan)

        for i in range(n_tournaments):
            winner = None
            winner_fitness = -1

            for j in range(n_candidates):
                fitness = pop[P[i, j]].X[0].get_fitness()
                if fitness > winner_fitness:
                    winner_fitness = fitness
                    winner = P[i, j]

            S[i] = winner

        return S[:, None].astype(int, copy=False)

    def __init__(self, tournament_size=2):
        super().__init__(func_comp=NaryTournamentSelectionOnFitness.nary_fitness_tournament, pressure=tournament_size)
