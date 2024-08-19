import copy

from pymoo.core.crossover import Crossover
import numpy as np


class PittsburghCrossover(Crossover):
    """Pittsburgh crossover

    Attributes:
        __min_num_rules (int): Min number of rules that the Pittsburgh solution must contain
        __max_num_rules (int): Max number of rules that the Pittsburgh solution can contain
        _random_gen (numpy.random.Generator): Random generator
    """
    def __init__(self, min_num_rules, max_num_rules, random_gen, prob=0.9):
        """Constructor

        Args:
            min_num_rules (int): Min number of rules that the Pittsburgh solution must contain
            max_num_rules (int): Max number of rules that the Pittsburgh solution can contain
            random_gen (numpy.random.Generator): Random generator
            prob (float): Crossover probability
        """
        super().__init__(2, 1, prob)
        self.__min_num_rules = min_num_rules
        self.__max_num_rules = max_num_rules
        self._random_gen = random_gen

        if self.__max_num_rules < 0:
            raise ValueError("The maximum number of rules can't be negative")

    def get_num_rules_from_parents(self, num_rules_p1, num_rules_p2):
        """Get the number of rules that are taken from each parent

        Args:
            num_rules_p1 (int): Number of rules in the 1st parent
            num_rules_p2 (int): Number of rules in the 2nd parent
        Returns:
            int: Number of rules taken from the 1st parent
            int: Number of rules taken from the 2nd parent
        """
        num_rules_from_p1 = self._random_gen.integers(0, num_rules_p1)
        num_rules_from_p2 = self._random_gen.integers(0, num_rules_p2)
        sum_num_rules = num_rules_from_p1 + num_rules_from_p2

        if sum_num_rules > self.__max_num_rules:
            # Remove rules excess
            num_deletions = sum_num_rules - self.__max_num_rules
            for j in range(num_deletions):
                if num_rules_from_p1 > 0 and num_rules_from_p2 > 0:
                    if self._random_gen.random() < 0.5:
                        num_rules_from_p1 -= 1
                    else:
                        num_rules_from_p2 -= 1
                elif num_rules_from_p1 == 0 and num_rules_from_p2 > 0:
                    num_rules_from_p2 -= 1
                elif num_rules_from_p2 == 0 and num_rules_from_p1 > 0:
                    num_rules_from_p1 -= 1
        elif sum_num_rules < self.__min_num_rules:
            # Add missing rules
            num_additions = self.__min_num_rules - sum_num_rules

            for j in range(num_additions):
                if num_rules_from_p1 < self.__max_num_rules and num_rules_from_p2 < self.__max_num_rules:
                    if self._random_gen.random() < 0.5:
                        num_rules_from_p1 += 1
                    else:
                        num_rules_from_p2 += 1
                elif num_rules_from_p1 == self.__max_num_rules and num_rules_from_p2 < self.__max_num_rules:
                    num_rules_from_p2 += 1
                elif num_rules_from_p2 == self.__max_num_rules and num_rules_from_p1 > self.__max_num_rules:
                    num_rules_from_p1 += 1
        return num_rules_from_p1, num_rules_from_p2

    def _do(self, problem, X, **kwargs):
        """Run the crossover on the given population

        Args:
            problem (Problem): Optimization problem (e.g. PittsburghProblem)
            X (object[,]): Population. The shape is (n_parents, n_matings, n_var),
            **kwargs (dict): Other arguments taken by Pymoo crossover object

        Returns:
            double[,,]: Crossover offspring. Shape: (1, n_matings, 1)
        """
        _, n_matings, _ = X.shape
        Y = np.zeros((1, n_matings, 1), dtype=object)

        for i in range(n_matings):
            p1 = X[0,i,0]
            p2 = X[1,i,0]

            Y[0,i,0] = copy.deepcopy(p1)


            Y[0,i,0].clear_vars()
            Y[0,i,0].clear_attributes()

            num_rules_from_p1, num_rules_from_p2 = self.get_num_rules_from_parents(p1.get_num_vars(), p2.get_num_vars())
            rules_idx_from_p1 = self._random_gen.choice(np.arange(p1.get_num_vars(), dtype=int), num_rules_from_p1, replace=False)
            rules_idx_from_p2 = self._random_gen.choice(np.arange(p2.get_num_vars(), dtype=int), num_rules_from_p2, replace=False)

            new_vars = np.empty(num_rules_from_p1 + num_rules_from_p2, dtype=object)
            j = 0
            for rule_idx in rules_idx_from_p1:
                new_vars[j] = copy.deepcopy(p1.get_var(rule_idx))
                j += 1
            for rule_idx in rules_idx_from_p2:
                new_vars[j] = copy.deepcopy(p2.get_var(rule_idx))
                j += 1

            Y[0, i, 0].set_vars(new_vars)
            if not Y[0, i, 0].are_rules_valid():
                print("WARNING: Invalid rule generated in Pittsburgh crossover")
        return Y

    def execute(self, problem, X, **kwargs):
        """Public version of the _do function (needed for the Hybrid crossover). Run the crossover on the given population

        Args:
            problem (Problem): Optimization problem (e.g. PittsburghProblem)
            X (object[,]): Population. The shape is (n_parents, n_matings, n_var),
            **kwargs (dict): Other arguments taken by Pymoo crossover object

        Returns:
            double[,,]: Crossover offspring. Shape: (1, n_matings, 1)
        """
        return self._do(problem, X, **kwargs)
