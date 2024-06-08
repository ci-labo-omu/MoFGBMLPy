import copy

from pymoo.core.crossover import Crossover


class UniformCrossover(Crossover):
    def __init__(self, prob=0.9):
        super().__init__(2, 1, prob)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.zeros((1, n_matings, n_var), dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]
            if random.random() < 0.5:
                p1_antecedent_indices = X[0, k, 0].get_antecedent().get_antecedent_indices()
                p2_antecedent_indices = X[1, k, 0].get_antecedent().get_antecedent_indices()

                child_antecedent_indices = []
                for i in range(len(p1_antecedent_indices)):
                    if random.random() < 0.5:
                        child_antecedent_indices.append(p2_antecedent_indices[i])
                    else:
                        child_antecedent_indices.append(p1_antecedent_indices[i])
                Y[0, k, 0] = RuleBasic(Antecedent(child_antecedent_indices), None)
            else:
                if random.random() < 0.5:
                    Y[0, k, 0] = copy.copy(a)
                else:
                    Y[0, k, 0] = copy.copy(b)
        return Y
