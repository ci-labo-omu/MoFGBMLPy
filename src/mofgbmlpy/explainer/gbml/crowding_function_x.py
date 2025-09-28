import numpy as np

from mofgbmlpy.explainer.gbml.fuzzy_sets_eliminate_duplicates import FuzzySetsEliminateDuplicates


class CrowdingFunctionX:
    @staticmethod
    def calc_crowding_distance(X, **kwargs):
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
        return self.calc_crowding_distance(X, **kwargs)
