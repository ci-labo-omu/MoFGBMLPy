import copy

from pymoo.core.problem import Problem
import numpy as np
from gbml.solution.pittsburgh_solution import PittsburghSolution
import time


class PittsburghProblem(Problem):
    __num_vars = 0
    __num_objectives = None
    __num_constraints = None
    __training_ds = None
    __michigan_solution_builder = None
    __classifier = None

    def __init__(self,
                 num_vars,
                 num_objectives,
                 num_constraints,
                 training_dataset,
                 michigan_solution_builder,
                 classifier):

        super().__init__(n_var=1, n_obj=num_objectives)  # 1 var because we consider one solution object
        self.__training_ds = training_dataset
        self.__num_vars = num_vars
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classifier = classifier

    def create_solution(self):
        pittsburgh_solution = PittsburghSolution(self.__num_vars,
                                                 self.__num_objectives,
                                                 self.__num_constraints,
                                                 copy.copy(self.__michigan_solution_builder),
                                                 copy.copy(self.__classifier))

        michigan_solutions = self.__michigan_solution_builder.create(self.__num_vars)
        pittsburgh_solution.set_vars(michigan_solutions)

        return pittsburgh_solution

    # def __evaluate_one(self, solution):
    #     print(type(solution))
    #     return [solution.get_error_rate(self.__training_ds), solution.get_num_vars()]

    def get_num_vars(self):
        return self.__num_vars

    def get_num_objectives(self):
        return self.__num_objectives

    def get_num_constraints(self):
        return self.__num_constraints

    def get_training_set(self):
        return self.__training_ds

    def get_rule_builder(self):
        self.__michigan_solution_builder.get_rule_builder()

    def _evaluate(self, X, out, *args, **kwargs):
        # vfunc = np.vectorize(self.__evaluate_one)
        # out["F"] = vfunc(X)
        #
        # print(out["F"])

        out["F"] = np.zeros((len(X), 2), dtype=np.float_)

        for i in range(len(X)):
            out["F"][i][0] = X[i, 0].get_error_rate(self.__training_ds)
            out["F"][i][1] = X[i, 0].get_num_vars()  # num rules
