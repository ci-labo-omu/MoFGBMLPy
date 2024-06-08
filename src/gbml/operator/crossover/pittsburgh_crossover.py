import copy

from pymoo.core.crossover import Crossover
import numpy as np
import random

from main.consts import Consts


class PittsburghCrossover(Crossover):
    def __init__(self, prob=0.9):
        super().__init__(2, 1, prob)

    def get_num_rules_from_parents(self, num_rules_p1, num_rules_p2, n_vars):
        num_rules_from_p1 = random.randint(0, num_rules_p1 - 1)
        num_rules_from_p2 = random.randint(0, num_rules_p2 - 1)
        sum_num_rules = num_rules_from_p1 + num_rules_from_p2

        if sum_num_rules > Consts.MAX_RULE_NUM:
            # Remove rules excess
            num_deletions = sum_num_rules - Consts.MAX_RULE_NUM
            for j in range(num_deletions):
                if num_rules_from_p1 > 0 and num_rules_from_p2 > 0:
                    if random.random() < 0.5:
                        num_rules_from_p1 -= 1
                    else:
                        num_rules_from_p2 -= 1
                elif num_rules_from_p1 == 0 and num_rules_from_p2 > 0:
                    num_rules_from_p2 -= 1
                elif num_rules_from_p2 == 0 and num_rules_from_p1 > 0:
                    num_rules_from_p1 -= 1
                else:
                    raise Exception("No more rules can be deleted. The Consts parameters might be invalid")
        elif sum_num_rules < Consts.MIN_RULE_NUM:
            # Add missing rules
            num_additions = Consts.MIN_RULE_NUM - sum_num_rules
            max_per_parent = min(Consts.MAX_RULE_NUM, n_vars)

            for j in range(num_additions):
                if num_rules_from_p1 < max_per_parent and num_rules_from_p2 < max_per_parent:
                    if random.random() < 0.5:
                        num_rules_from_p1 += 1
                    else:
                        num_rules_from_p2 += 1
                elif num_rules_from_p1 == max_per_parent and num_rules_from_p2 < max_per_parent:
                    num_rules_from_p2 += 1
                elif num_rules_from_p2 == max_per_parent and num_rules_from_p1 > max_per_parent:
                    num_rules_from_p1 += 1
                else:
                    raise Exception("No more rules can be added. The Consts parameters might be invalid")
        return num_rules_from_p1, num_rules_from_p2

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings = X.shape
        Y = np.zeros((1, n_matings, 1), dtype=object)
        # we don't have to compute random() < prob since it's done in the parent class do method
        for i in range(n_matings):
            p1 = X[0][i]
            p2 = X[1][i]

            Y[0,i,0] = copy.copy(p1)
            Y[0,i,0].clear_vars()
            Y[0,i,0].clear_attributes()

            num_rules_from_p1, num_rules_from_p2 = self.get_num_rules_from_parents(p1.get_num_vars(), p2.get_num_vars(), problem.n_var)
            rules_idx_from_p1 = np.random.choice(list(range(p1.get_num_vars())), num_rules_from_p1, replace=False)
            rules_idx_from_p2 = np.random.choice(list(range(p2.get_num_vars())), num_rules_from_p2, replace=False)

            for rule_idx in rules_idx_from_p1:
                Y[0,i,0].add_var(copy.copy(p1.get_var(rule_idx)))
            for rule_idx in rules_idx_from_p2:
                Y[0,i,0].add_var(copy.copy(p2.get_var(rule_idx)))

        return Y
