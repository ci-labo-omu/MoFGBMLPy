import numpy as np
from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class NumWins(ObjectiveFunction):
    def __init__(self, data_set):
        self.__data_set = data_set

    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        cdef int i = 0
        cdef MichiganSolution sol

        if isinstance(solutions[0], MichiganSolution):
            # For each pattern, get the winner rule (highest fitness value)
            winner_rules_indices = np.empty(len(solutions), dtype=np.int_)
            winner_rules_fitness = np.full(len(solutions), fill_value=-1)

            for i in range(len(self.__data_set)):
                for j in range(len(solutions)):
                    sol = solutions[j]
                    current_fitness = sol.get_fitness_value(self.__data_set[i])
                    if current_fitness > winner_rules_fitness[i]:
                        winner_rules_fitness[j] = current_fitness
                        winner_rules_indices[i] = j

            for i in range(len(self.__data_set)):
                out[winner_rules_indices[i]] += 1

            for i in range(len(solutions)):
                sol = solutions[i]
                sol.set_objective(obj_index, out[i])
        else:
            raise Exception("Solution must be of type PittsburghSolution")

    def __repr__(self):
        return "Number of wins"