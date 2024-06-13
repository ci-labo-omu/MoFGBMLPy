import copy

from pymoo.core.problem import Problem
import numpy as np
from src.gbml.solution.pittsburgh_solution import PittsburghSolution
import time

class PittsburghProblem(Problem):
    __winner_solution_for_each_pattern = None
    __num_vars_pittsburgh = None
    __num_objectives_pittsburgh = None
    __num_constraints_pittsburgh = None
    __training_ds = None
    __michigan_solution_builder = None
    __classifier = None
    __num_vars = 0

    class WinnerSolution:
        __max_fitness_value = None
        __solution_index = None

        def __init__(self, max_fitness_value=None, __solution_index=None):
            self.__max_fitness_value = max_fitness_value
            self.__solution_index = __solution_index

        def get_max_fitness_value(self):
            return self.__max_fitness_value

        def get_solution_index(self):
            return self.__solution_index

    def __init__(self,
                 num_vars,
                 num_objectives,
                 num_constraints,
                 training_dataset,
                 michigan_solution_builder,
                 classifier):

        super().__init__(n_var=1, n_obj=num_objectives)  # 1 var because we consider one solution object
        self.__training_ds = training_dataset

        self.__winner_solution_for_each_pattern = np.zeros(training_dataset.get_size(), dtype=object)
        for i in range(training_dataset.get_size()):
            self.__winner_solution_for_each_pattern[i] = PittsburghProblem.WinnerSolution()

        self.__num_vars = num_vars
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classifier = classifier

    def create_solution(self):
        pittsburgh_solution = PittsburghSolution(self.__num_vars,
                                                 self.__num_objectives_pittsburgh,
                                                 self.__num_constraints_pittsburgh,
                                                 copy.copy(self.__michigan_solution_builder),
                                                 copy.copy(self.__classifier))


        michigan_solutions = self.__michigan_solution_builder.create(self.__num_vars)
        for solution in michigan_solutions:
            pittsburgh_solution.add_var(solution)

        return pittsburgh_solution

    # def __evaluate_one(self, solution):
    #     print(type(solution))
    #     return [solution.get_error_rate(self.__training_ds), solution.get_num_vars()]

    def _evaluate(self, X, out, *args, **kwargs):
        # vfunc = np.vectorize(self.__evaluate_one)
        # out["F"] = vfunc(X)
        #
        # print(out["F"])

        out["F"] = np.zeros((len(X), 2), dtype=np.float_)
        for i in range(len(X)):
            out["F"][i][0] = X[i, 0].get_error_rate(self.__training_ds)
            out["F"][i][1] = X[i, 0].get_num_vars()  # num rules
