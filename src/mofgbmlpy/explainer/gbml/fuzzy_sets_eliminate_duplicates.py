from pymoo.core.duplicate import DuplicateElimination
import numpy as np
from pymoo.core.population import Population

from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem


class FuzzySetsEliminateDuplicates(DuplicateElimination):
    """Duplicate elimination method to eliminate solutions with the same fuzzy sets parameters for the rules, by comparing the parameters of the fuzzy sets of the rules and eliminating those that are too similar.

    Attributes:
        epsilon (float): The threshold for considering two solutions as duplicates, based on the distance between their fuzzy sets parameters
        _problem (CounterfactualProblem): The counterfactual problem being solved, used to get the initial fuzzy sets if needed
    """

    def __init__(self, problem, epsilon=1e-16, **kwargs) -> None:
        """Constructor

        Args:
            problem (CounterfactualProblem): The counterfactual problem being solved, used to get the initial fuzzy sets if needed
            epsilon (float, optional): The threshold for considering two solutions as duplicates, based on the distance between their fuzzy sets parameters. Defaults to 1e-16.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self._problem = problem

    @staticmethod
    def distance_mfs_params_count_differences(params_1, params_2, threshold=1e-8):
        """Calculate the distance between two sets of fuzzy sets parameters by counting the number of differences in the parameters, considering two parameters as different if their absolute difference is greater than a given threshold.

        Args:
            params_1 (list): The first set of fuzzy sets parameters, where each element is an array of parameters for a fuzzy set
            params_2 (list): The second set of fuzzy sets parameters, where each element is an array of parameters for a fuzzy set
            threshold (float, optional): The threshold for considering two parameters as different. Defaults to 1e-8.

        Returns:
            int: The distance between the two sets of fuzzy sets parameters, calculated as the number of differences in the parameters
        """
        if len(params_1) != len(params_2):
            return max(len(params_1), len(params_2))

        distance = 0
        for fs_i in range(len(params_1)):
            if len(params_1[fs_i]) != len(params_2[fs_i]):
                distance += 1
            elif len(params_1[fs_i]) != 0:
                for i in range(len(params_1[fs_i])):
                    distance += 1 if abs(params_1[fs_i][i] - params_2[fs_i][i]) > threshold else 0

        return distance

    @staticmethod
    def distance_mfs_params(params_1, params_2):
        """Calculate the distance between two sets of fuzzy sets parameters by calculating the average absolute difference between the parameters, considering two parameters as different if their absolute difference is greater than a given threshold.

        Args:
            params_1 (list): The first set of fuzzy sets parameters, where each element is an array of parameters for a fuzzy set
            params_2 (list): The second set of fuzzy sets parameters, where each element is an array of parameters for a fuzzy set

        Returns:
            float: The distance between the two sets of fuzzy sets parameters, calculated as the average absolute difference between the parameters
        """
        if len(params_1) != len(params_2):
            return 1

        distance = 0
        for fs_i in range(len(params_1)):
            if len(params_1[fs_i]) != len(params_2[fs_i]):
                distance += 1
            elif len(params_1[fs_i]) != 0:
                dist_fs = 0
                for i in range(len(params_1[fs_i])):
                    dist_fs += abs(params_1[fs_i][i] - params_2[fs_i][i])
                distance += dist_fs / len(params_1[fs_i])

        return distance / len(params_1)

    @staticmethod
    def extract_params(pop):
        """Extract the parameters of the fuzzy sets of the rules from a population of solutions, where each solution is expected to have a rule with fuzzy sets as antecedents.

        Args:
            pop (Population or np.ndarray): The population of solutions, which can be a pymoo Population or a numpy array of solutions. Each solution is expected to have a rule with fuzzy sets as antecedents.
        Returns:
            np.ndarray: An array of shape (n_solutions, n_fuzzy_sets) containing the parameters of the fuzzy sets of the rules for each solution in the population, where n_solutions is the number of solutions in the population and n_fuzzy_sets is the number of fuzzy sets in the rules.
        """
        if not isinstance(pop, Population):
            if not isinstance(pop, np.ndarray):
                raise ValueError("pop must be a numpy array or be a pymoo Population")
            X = pop
        else:
            X = pop.get("X")

        params_x = np.empty((len(X), X[0][0].get_rule().get_antecedent_array_size()), dtype=object)
        for i in range(len(X)):
            fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(X[i][0].get_rule())
            for j in range(len(fuzzy_sets)):
                params_x[i][j] = fuzzy_sets[j].get_function().get_params()
        return params_x

    @staticmethod
    def calc_dist_count_differences(pop):
        """Calculate the distance between the solutions in a population by counting the number of differences in the parameters of the fuzzy sets of the rules, using the distance_mfs_params_count_differences method.

        Args:
            pop (Population or np.ndarray): The population of solutions, which can be a pymoo Population or a numpy array of solutions. Each solution is expected to have a rule with fuzzy sets as antecedents.

        Returns:
            np.ndarray: An array of shape (n_solutions, n_solutions) containing the distances between the solutions in the population, where n_solutions is the number of solutions in the population. The distance between two solutions is calculated as the number of differences in the parameters of the fuzzy sets of their rules, using the distance_mfs_params_count_differences method.
        """
        distance = np.empty((len(pop), len(pop)))
        params_x = FuzzySetsEliminateDuplicates.extract_params(pop)

        for i in range(distance.shape[0]):
            for j in range(i, distance.shape[1]):
                if i == j:
                    distance[i][j] = 0
                    continue
                distance[i][j] = FuzzySetsEliminateDuplicates.distance_mfs_params_count_differences(
                    params_x[i], params_x[j]
                )
                distance[j][i] = distance[i][j]

        return distance

    @staticmethod
    def calc_dist(pop):
        """Calculate the distance between the solutions in a population by calculating the average absolute difference between the parameters of the fuzzy sets of the rules, using the distance_mfs_params method.

        Args:
            pop (Population or np.ndarray): The population of solutions, which can be a pymoo Population or a numpy array of solutions. Each solution is expected to have a rule with fuzzy sets as antecedents.

        Returns:
            np.ndarray: An array of shape (n_solutions, n_solutions) containing the distances between the solutions in the population, where n_solutions is the number of solutions in the population. The distance between two solutions is calculated as the average absolute difference between the parameters of the fuzzy sets of their rules, using the distance_mfs_params method.
        """
        distance = np.empty((len(pop), len(pop)))
        params_x = FuzzySetsEliminateDuplicates.extract_params(pop)

        for i in range(distance.shape[0]):
            for j in range(i, distance.shape[1]):
                if i == j:
                    distance[i][j] = 0
                    continue
                distance[i][j] = FuzzySetsEliminateDuplicates.distance_mfs_params(params_x[i], params_x[j])
                distance[j][i] = distance[i][j]

        return distance

    def is_duplicate_in_other_pop(self, rule, other):
        """Check if a solution with the same fuzzy sets parameters for the rules already exists in another population, by comparing the parameters of the fuzzy sets of the rules and checking if they are too similar.

        Args:
            rule (Rule): The rule of the solution to check for duplicates, which is expected to have fuzzy sets as antecedents.
            other (Population or np.ndarray): The other population of solutions to check for duplicates, which can be a pymoo Population or a numpy array of solutions. Each solution in the other population is expected to have a rule with fuzzy sets as antecedents.

        Returns:
            bool: True if a solution with the same fuzzy sets parameters for the rules already exists in the other population, False otherwise.
        """
        rule_fs = CounterfactualProblem.get_fuzzy_sets_from_rule(rule)
        rule_params = np.empty(len(rule_fs), dtype=object)
        for k in range(len(rule_fs)):
            rule_params[k] = rule_fs[k].get_function().get_params()
        for j in range(len(other)):
            other_fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(other[j].X[0].get_rule())
            params_other = np.empty(len(other_fuzzy_sets), dtype=object)
            for k in range(len(other_fuzzy_sets)):
                params_other[k] = other_fuzzy_sets[k].get_function().get_params()

            if (
                FuzzySetsEliminateDuplicates.distance_mfs_params_count_differences(params_other, rule_params)
                < self.epsilon
            ):
                return True

    def _do(self, pop, other, is_duplicate):
        """Eliminate solutions with the same fuzzy sets parameters for the rules from a population, by comparing the parameters of the fuzzy sets of the rules and eliminating those that are too similar.

        Args:
            pop (Population or np.ndarray): The population of solutions to eliminate duplicates from, which can be a pymoo Population or a numpy array of solutions. Each solution is expected to have a rule with fuzzy sets as antecedents.
            other (Population or np.ndarray): The other population of solutions to check for duplicates, which can be a pymoo Population or a numpy array of solutions. Each solution in the other population is expected to have a rule with fuzzy sets as antecedents.
            is_duplicate (np.ndarray): An array of shape (n_solutions,) containing boolean values indicating whether each solution in the population is a duplicate or not. This array will be updated by this method to mark the solutions that are considered duplicates based on the distance between their fuzzy sets parameters and the parameters of the solutions in the other population.

        Returns:
            np.ndarray: An array of shape (n_solutions,) containing boolean values indicating whether each solution in the population is a duplicate or not, after eliminating the duplicates based on the distance between their fuzzy sets parameters and the parameters of the solutions in the other population.
        """
        distance = self.calc_dist_count_differences(pop)
        n = len(distance)
        is_duplicate = np.zeros(n, dtype=bool)

        for i in range(n):
            # distance between current solutions
            if np.any(distance[i, :i] < self.epsilon):
                is_duplicate[i] = True

            # distance to previous population
            if other is not None:
                is_duplicate[i] = self.is_duplicate_in_other_pop(pop[i].X[0].get_rule(), other)

        return is_duplicate
