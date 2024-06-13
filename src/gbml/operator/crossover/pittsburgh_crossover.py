import copy

from pymoo.core.crossover import Crossover
import numpy as np
import random

import time
import concurrent.futures


class PittsburghCrossover(Crossover):
    __min_num_rules = None
    __max_num_rules = None

    def __init__(self, min_num_rules, max_num_rules, prob=0.9):
        super().__init__(2, 1, prob)
        self.__min_num_rules = min_num_rules
        self.__max_num_rules = max_num_rules

    def get_num_rules_from_parents(self, num_rules_p1, num_rules_p2, n_vars):
        num_rules_from_p1 = random.randint(0, num_rules_p1 - 1)
        num_rules_from_p2 = random.randint(0, num_rules_p2 - 1)
        sum_num_rules = num_rules_from_p1 + num_rules_from_p2

        if sum_num_rules > self.__max_num_rules:
            # Remove rules excess
            num_deletions = sum_num_rules - self.__max_num_rules
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
                    raise Exception("No more rules can be deleted.")
        elif sum_num_rules < self.__min_num_rules:
            # Add missing rules
            num_additions = self.__min_num_rules - sum_num_rules
            max_per_parent = min(self.__max_num_rules, n_vars)

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
                    raise Exception("No more rules can be added")
        return num_rules_from_p1, num_rules_from_p2

    # def do_single_crossover(self, parent1, parent2, n_var):
    #     p1 = parents[0]
    #     p2 = parents[1]
    #
    #     offspring = copy.copy(p1)
    #     offspring.clear_vars()
    #     offspring.clear_attributes()
    #
    #     num_rules_from_p1, num_rules_from_p2 = self.get_num_rules_from_parents(p1.get_num_vars(), p2.get_num_vars(),
    #                                                                            n_var)
    #     rules_idx_from_p1 = np.random.choice(list(range(p1.get_num_vars())), num_rules_from_p1, replace=False)
    #     rules_idx_from_p2 = np.random.choice(list(range(p2.get_num_vars())), num_rules_from_p2, replace=False)
    #
    #     for rule_idx in rules_idx_from_p1:
    #         offspring.add_var(copy.copy(p1.get_var(rule_idx)))
    #     for rule_idx in rules_idx_from_p2:
    #         offspring.add_var(copy.copy(p2.get_var(rule_idx)))
    #     return offspring

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.zeros((1, n_matings, 1), dtype=object)

        start = time.time()

        for i in range(n_matings):
            p1 = X[0,i,0]
            p2 = X[1,i,0]

            Y[0,i] = copy.copy(p1)
            Y[0,i,0].clear_vars()
            Y[0,i,0].clear_attributes()

            num_rules_from_p1, num_rules_from_p2 = self.get_num_rules_from_parents(p1.get_num_vars(), p2.get_num_vars(), n_var)
            rules_idx_from_p1 = np.random.choice(list(range(p1.get_num_vars())), num_rules_from_p1, replace=False)
            rules_idx_from_p2 = np.random.choice(list(range(p2.get_num_vars())), num_rules_from_p2, replace=False)

            for rule_idx in rules_idx_from_p1:
                Y[0,i,0].add_var(copy.copy(p1.get_var(rule_idx)))
            for rule_idx in rules_idx_from_p2:
                Y[0,i,0].add_var(copy.copy(p2.get_var(rule_idx)))

        elapsed = time.time() - start  # 11.96s
        return Y
