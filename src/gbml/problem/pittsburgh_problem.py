from pymoo.core.problem import Problem
import numpy as np


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

        super().__init__(n_var=num_vars, n_obj=num_objectives)
        self.__training_ds = training_dataset
        self.__winner_solution_for_each_pattern = np.array(
            [PittsburghProblem.WinnerSolution()] * training_dataset.get_size(), dtype=object)
        self.__num_vars = num_vars
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classifier = classifier

    def create_solution(self):
        pittsburgh_solution = PittsburghSolution(self.__num_vars,
                                                 self.__num_objectives_pittsburgh,
                                                 self.__num_constraints_pittsburgh,
                                                 self.__michigan_solution_builder.copy(),
                                                 self.__classifier.copy())

        michigan_solutions = self.__michigan_solution_builder.create_michigan_solution(self.__num_vars)
        for solution in michigan_solutions:
            pittsburgh_solution.add_var(solution)

        return pittsburgh_solution

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.zeros((len(X), 2), dtype=np.float_)
        patterns = self.__training_ds.get_patterns()

        for i in range(len(X)):
            out["F"][i][0] = 0
            out["F"][i][1] = X[i].get_rule_length()

            if X[i, 0].is_rejected_class_label():
                out["F"][i][0] = -1  # This rule must be deleted during environmental selection
                continue

            for j in range(len(patterns)):
                fitness_value = X[i].get_fitness_value(patterns[j].get_attribute_vector())
                max_fitness_value = self.__winner_solution_for_each_pattern[j].get_max_fitness_value()
                if max_fitness_value is None or fitness_value > max_fitness_value:
                    self.__winner_solution_for_each_pattern[j] = PittsburghProblem.WinnerSolution(
                        -fitness_value, i)

        for i in range(len(patterns)):
            out["F"][self.__winner_solution_for_each_pattern[i].get_solution_index()][0] -= 1
