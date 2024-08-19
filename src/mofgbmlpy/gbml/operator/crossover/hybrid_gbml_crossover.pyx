import copy

from pymoo.core.crossover import Crossover
import numpy as np

class HybridGBMLCrossover(Crossover):
    """Hybrid crossover between Michigan and Pittsburgh crossovers

    Attributes:
        _random_gen (numpy.random.Generator): Random generator
        __michigan_crossover_probability (float): Probability that a Michigan crossover occurs instead of a Pittsburgh one
        __michigan_crossover (MichiganCrossover): Probability that a Michigan crossover occurs after it has been decided that the crossover type would be the Michigan one
        __pittsburgh_crossover (PittsburghCrossover): Pittsburgh crossover used here depending on the Michigan crossover probability
    """
    def __init__(self, random_gen, michigan_crossover_probability, michigan_crossover, pittsburgh_crossover, prob=0.9):
        """Constructor

        Args:
            random_gen (numpy.random.Generator): Random generator
            michigan_crossover_probability (float): Probability that a Michigan crossover occurs instead of a Pittsburgh one
            michigan_crossover (MichiganCrossover): Probability that a Michigan crossover occurs after it has been decided that the crossover type would be the Michigan one
            pittsburgh_crossover (PittsburghCrossover): Pittsburgh crossover used here depending on the Michigan crossover probability
            prob (float): Probability that a crossover occurs
        """
        super().__init__(2, 1, prob)
        self._random_gen = random_gen
        self.__michigan_crossover_probability = michigan_crossover_probability
        self.__michigan_crossover = michigan_crossover
        self.__pittsburgh_crossover = pittsburgh_crossover

    def _do(self, problem, X, **kwargs):
        """Run the crossover on the given population

        Args:
            problem (Problem): Optimization problem (e.g. PittsburghProblem)
            X (object[,,]): Population. The shape is (2, n_matings, n_var),
            **kwargs (dict): Other arguments taken by Pymoo crossover object

        Returns:
            double[,,]: Crossover offspring. Shape: (1, n_matings, 1)
        """
        _, n_matings, n_var = X.shape

        michigan_crossover_mask = np.empty(n_matings, dtype=np.bool_)

        cdef bint at_least_one_michigan_crossover = False
        cdef bint at_least_one_pittsburgh_crossover = False

        for i in range(n_matings):
            # Check if the 2 parents are identical
            if X[0,i,0] == X[1,i,0]:
                michigan_crossover_mask[i] = True
                if not at_least_one_michigan_crossover:
                    at_least_one_michigan_crossover = True
            else:
                michigan_crossover_mask[i] = self._random_gen.random() < self.__michigan_crossover_probability
                if not at_least_one_michigan_crossover and michigan_crossover_mask[i]:
                    at_least_one_michigan_crossover = True
                elif not at_least_one_pittsburgh_crossover and not michigan_crossover_mask[i]:
                    at_least_one_pittsburgh_crossover = True

        if at_least_one_pittsburgh_crossover:
            Y_pittsburgh = self.__pittsburgh_crossover.execute(problem, X[:, np.invert(michigan_crossover_mask)], **kwargs)

        if at_least_one_michigan_crossover:
            Y_michigan = self.__michigan_crossover.execute(problem, X[0, michigan_crossover_mask], **kwargs)

        if at_least_one_michigan_crossover and at_least_one_pittsburgh_crossover:
            return np.concatenate((Y_michigan, Y_pittsburgh), axis=1)
        elif not at_least_one_michigan_crossover:
            return Y_pittsburgh
        elif not at_least_one_pittsburgh_crossover:
            return Y_michigan
        else:
            raise ValueError("No offspring created during hybrid crossover. It might be because n_matings is null")
