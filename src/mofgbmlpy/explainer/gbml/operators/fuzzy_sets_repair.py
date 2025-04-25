from pymoo.core.repair import Repair


class FuzzySetsRepair(Repair):
    def _do(self, problem, X, **kwargs):
        repaired_x = X.copy()

        for i in range(len(X)):
            j = 0
            fuzzy_set_index = 0

            while j < problem.n_var:
                fuzzy_set_index_size = problem.get_fuzzy_set_size(fuzzy_set_index)
                if fuzzy_set_index_size == 3:  # triangular
                    repaired_x[i][j] = max(0, min(1, repaired_x[i][j]))
                    repaired_x[i][j + 1] = max(0, min(repaired_x[i][j], repaired_x[i][j + 1]))
                    repaired_x[i][j + 2] = max(0, min(1, repaired_x[i][j + 2]))

                    j += 2
                    fuzzy_set_index += 1

                j += 1

        return repaired_x
