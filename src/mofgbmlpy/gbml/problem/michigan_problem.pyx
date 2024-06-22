import copy

from pymoo.core.problem import Problem
import numpy as np

from mofgbmlpy.fuzzy.classifier.classifier import Classifier
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution


class MichiganProblem(Problem):
    __num_vars = 0
    __num_objectives = None
    __num_constraints = None
    __training_ds = None
    __rule_builder = None

    def __init__(self,
                 num_vars,
                 num_objectives,
                 num_constraints,
                 training_dataset,
                 rule_builder):

        super().__init__(n_var=1, n_obj=num_objectives)  # 1 var because we consider one solution object
        self.__training_ds = training_dataset
        self.__num_vars = num_vars
        self.__rule_builder = rule_builder

    # def create_solution(self):
    #     michigan_solution = MichiganSolution(self.__num_vars,
    #                                          self.__num_objectives_pittsburgh,
    #                                          self.__num_constraints_pittsburgh,
    #                                          copy.copy(self.__rule_builder))
    #
    #     return michigan_solution

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.zeros((len(X), 2), dtype=np.float64)

        # For each pattern, get the winner rule (highest fitness value)
        winner_rules_indices = np.empty(len(X), dtype=np.int_)
        winner_rules_fitness = np.full(len(X), fill_value=-1)
        for i in range(len(self.__training_ds)):
            for j in range(len(X)):
                current_fitness = X[i, 0].get_fitness_value(X)
                if current_fitness > winner_rules_fitness[i]:
                    winner_rules_fitness[j] = current_fitness
                    winner_rules_indices[i] = j

        # Objective 1
        for i in range(len(self.__training_ds)):
            out["F"][winner_rules_indices[i]][0] += 1

        # Objective 2
        for i in range(len(X)):
            out["F"][i][1] = X[i, 0].get_rule_length()  # num rules
