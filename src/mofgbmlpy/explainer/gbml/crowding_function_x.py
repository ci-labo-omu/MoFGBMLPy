import numpy as np

from mofgbmlpy.explainer.gbml.fuzzy_sets_eliminate_duplicates import FuzzySetsEliminateDuplicates


class CrowdingFunctionX:
    """Crowding function to calculate the crowding distance of solutions based on the distance between their fuzzy sets, used to eliminate duplicates in the non-dominated solutions."""

    @staticmethod
    def calc_crowding_distance(X, **kwargs):
        """Calculate the crowding distance of solutions based on the distance between their fuzzy sets, used to eliminate duplicates in the non-dominated solutions.

        Args:
            X (np.ndarray): An array of shape (n_solutions, 1) containing the solutions with fuzzy sets for the rules, where each solution is an object with a get_rule() method that returns a rule object with a get_antecedent_array_size() method and a get_antecedent_array() method that returns an array of fuzzy sets for the antecedents of the rule.

        Returns:
            np.ndarray: An array of shape (n_solutions,) containing the crowding distance of each solution, calculated as the sum of the distances to the two nearest neighbors in the fuzzy sets space.
        """
        n_points, _ = X.shape
        distances = FuzzySetsEliminateDuplicates.calc_dist(X)

        # Get the two nearest neighbors for each point
        cd = np.zeros(n_points)
        for i in range(n_points):
            i_min_1 = None
            i_min_2 = None
            for j in range(n_points):
                if i != j:
                    if i_min_1 is None or distances[i][j] < distances[i][i_min_1]:
                        i_min_2 = i_min_1
                        i_min_1 = j
                    elif i_min_2 is None or distances[i][j] < distances[i][i_min_2]:
                        i_min_2 = j
            if i_min_1 is not None:
                cd[i] += distances[i][i_min_1]
            if i_min_2 is not None:
                cd[i] += distances[i][i_min_2]

        return cd

    def do(self, X, **kwargs):
        """Calculate the crowding distance of solutions based on the distance between their fuzzy sets, used to eliminate duplicates in the non-dominated solutions.

        Args:
            X (np.ndarray): An array of shape (n_solutions, 1) containing the solutions with fuzzy sets for the rules, where each solution is an object with a get_rule() method that returns a rule object with a get_antecedent_array_size() method and a get_antecedent_array() method that returns an array of fuzzy sets for the antecedents of the rule.

        Returns:
            np.ndarray: An array of shape (n_solutions,) containing the crowding distance of each solution, calculated as the sum of the distances to the two nearest neighbors in the fuzzy sets space.
        """
        return self.calc_crowding_distance(X, **kwargs)
