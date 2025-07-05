import numpy as np
from pymoo.operators.selection.tournament import Selection


class NaryTournamentSelectionOnFitness(Selection):
    """N-ary tournament selection operator for Michigan solutions based on fitness. Used to select parents"""
    @staticmethod
    def nary_fitness_tournament(pop, P, **kwargs):
        """

        Args:
            pop (Population): Population from where parents are selected
            P (Population): indices of the candidates in the tournaments. Shape: (n_tournaments, n_candidates)
            **kwargs (dict): Other arguments for Pymoo

        Returns:
            int[]: Array of size n_tournaments of the winners of the tournaments (index of individuals in the population)
        """
        n_tournaments, n_candidates = P.shape

        if n_candidates < 0:
            raise ValueError("tournament_size must be positive")
        elif n_candidates == 1:
            return P # TODO: test this conditional branch

        S = np.empty(n_tournaments, dtype=int)

        for i in range(n_tournaments):
            winner = None
            winner_fitness = -1

            for j in range(n_candidates):
                fitness = pop[P[i, j]].X[0].get_fitness()
                if fitness > winner_fitness:
                    winner_fitness = fitness
                    winner = P[i, j]

            S[i] = winner
        return S[:, None]

    def __init__(self, random_gen, tournament_size=2, **kwargs):
        """Constructor

        Args:
            tournament_size (int): Size of the tournament
        """
        super().__init__(**kwargs)

        # selection pressure to be applied
        self.pressure = tournament_size
        self.func_comp = NaryTournamentSelectionOnFitness.nary_fitness_tournament
        self._random_gen = random_gen

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        mating_pool_size = n_select * n_parents
        P = np.empty((mating_pool_size, self.pressure), dtype=int)
        for i in range(mating_pool_size):
            tournament = self._random_gen.choice(len(pop), size=self.pressure, replace=False)
            # tournament = set()
            # while len(tournament) < self.pressure:
            #     tournament.add(self._random_gen.integers(len(pop)))
            P[i] = np.array(list(tournament), dtype=int)

        S = self.func_comp(pop, P, **kwargs)
        return np.reshape(S, (n_select, n_parents))
