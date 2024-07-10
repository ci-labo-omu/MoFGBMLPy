import copy

from pymoo.core.problem import Problem
import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.classifier.classifier import Classifier
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution


class MichiganProblem(Problem):
    __num_vars = 0
    __objectives = None
    __num_constraints = None
    __training_ds = None
    __rule_builder = None

    def __init__(self,
                 num_vars,
                 objectives,
                 num_constraints,
                 training_dataset,
                 rule_builder):

        super().__init__(n_var=1, n_obj=len(objectives))  # 1 var because we consider one solution object
        self.__training_ds = training_dataset
        self.__num_vars = num_vars
        self.__rule_builder = rule_builder
        self.__objectives = objectives

    # def create_solution(self):
    #     michigan_solution = MichiganSolution(self.__num_vars,
    #                                          self.__num_objectives_pittsburgh,
    #                                          self.__num_constraints_pittsburgh,
    #                                          copy.copy(self.__rule_builder))
    #
    #     return michigan_solution

    def _evaluate(self, X, out, *args, **kwargs):
        cdef cnp.ndarray[double, ndim=2] eval_values = np.empty((len(X), self.get_num_objectives()), dtype=np.float64)
        cdef int i

        for i in range(len(self.__objectives)):
            self.__objectives[i].run(X[:, 0], i, eval_values[:, i])
        out["F"] = eval_values

    def get_num_objectives(self):
        return len(self.__objectives)
