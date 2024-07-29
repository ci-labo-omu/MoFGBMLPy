import sys
import copy

from pymoo.core.population import Population
from pymoo.core.problem import Problem
import numpy as np
cimport numpy as cnp
from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution
import time
import cython

class PittsburghProblem(Problem):
    __num_vars = 0
    __num_constraints = None
    __training_ds = None
    __michigan_solution_builder = None
    __classifier = None
    __objectives = None

    def __init__(self,
                 num_vars,
                 objectives,
                 num_constraints,
                 training_dataset,
                 michigan_solution_builder,
                 classifier):

        super().__init__(n_var=1, n_obj=len(objectives))  # 1 var because we consider one solution object
        self.__training_ds = training_dataset
        self.__num_vars = num_vars
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classifier = classifier
        self.__objectives = objectives
        self.__num_constraints = num_constraints
        if len(objectives) == 0:
            raise Exception("At least one objective is needed")

    def create_solution(self):
        pittsburgh_solution = PittsburghSolution(self.__num_vars,
                                                 self.get_num_objectives(),
                                                 self.__num_constraints,
                                                 copy.deepcopy(self.__michigan_solution_builder),
                                                 self.__classifier)

        return pittsburgh_solution

    def get_num_vars(self):
        return self.__num_vars

    def get_num_objectives(self):
        return len(self.__objectives)

    def get_num_constraints(self):
        return self.__num_constraints

    def get_training_set(self):
        return self.__training_ds

    def get_rule_builder(self):
        self.__michigan_solution_builder.get_rule_builder()

    def _evaluate(self, X, out, *args, **kwargs):
        cdef cnp.ndarray[double, ndim=2] eval_values = np.empty((len(X), self.get_num_objectives()), dtype=np.float64)
        cdef int i

        for i in range(len(self.__objectives)):
            self.__objectives[i].run(X[:, 0], i, eval_values[:,i])
        out["F"] = eval_values


    def remove_no_winner_michigan_solution(self, solutions):
        for i in range(len(solutions)):
            sol = solutions[i][0]
            num_vars = sol.get_num_vars()

            # Update eval values
            sol.get_error_rate(self.__training_ds)

            k = 0
            r = num_vars
            for j in range(num_vars):
                # if r == 1:
                #     break # Only one solution
                if sol.get_var(k).get_num_wins() < 1:
                    sol.remove_var(k)
                    r -= 1
                else:
                    k += 1

            if sol.get_num_vars() == 0:
                raise Exception("No michigan solution is remaining")

        return solutions