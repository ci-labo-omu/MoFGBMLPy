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

        cdef cnp.ndarray[double, ndim=2] eval_values = np.empty((len(X), 2), dtype=np.float64)
        cdef int i

        for i in range(len(X)):
            eval_values[i][0] = X[i, 0].get_error_rate(self.__training_ds)
            eval_values[i][1] = X[i, 0].get_num_vars()  # num rules

        out["F"] = eval_values

    @staticmethod
    def remove_no_winner_michigan_solution(individuals):
        # new_pop = []

        for i in range(len(individuals)):
            sol = individuals[i].X[0]
            num_vars = sol.get_num_vars()

            k = 0
            for j in range(num_vars):
                if sol.get_var(k).get_num_wins() < 1:
                    sol.remove_var(k)
                else:
                    k += 1

            if sol.get_num_vars() == 0:
                raise Exception("No michigan solution is remaining")

            # if sol.get_num_vars() != 0:
            #     new_pop.append(individuals[i])
        #
        # print(len(new_pop))
        # for x in new_pop:
        #     print("\t", x.X[0].get_num_vars())

        # return Population(new_pop)
        return individuals

    def remove_no_winner_michigan_solution2(self, solutions):
        for i in range(len(solutions)):
            sol = solutions[i][0]
            num_vars = sol.get_num_vars()

            # Update eval values
            sol.get_error_rate(self.__training_ds)

            k = 0
            r = num_vars
            for j in range(num_vars):
                if r == 1:
                    break # Only one solution # TODO: check if this is valid (not in the Java version but issue with AllCombinationsFactory
                # print(j, num_vars, sol.get_var(k).get_num_wins())
                if sol.get_var(k).get_num_wins() < 1:
                    sol.remove_var(k)
                    r -= 1
                else:
                    k += 1

            if sol.get_num_vars() == 0:
                raise Exception("No michigan solution is remaining")

        return solutions