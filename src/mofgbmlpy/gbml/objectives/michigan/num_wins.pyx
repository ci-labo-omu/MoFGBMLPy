import numpy as np

from mofgbmlpy.exception.invalid_solution_type_exception import InvalidSolutionTypeException
from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class NumWins(ObjectiveFunction):
    """Objective function that counts the number of times a solutions wins. For each pattern the winner is the solution with the highest fitness value

    Attributes:
        __data_set (Dataset): Training dataset used to compute the number of wins for each michigan solution
    """
    def __init__(self, data_set):
        """Constructor

        Args:
            data_set (Dataset): Training dataset used to compute the number of wins for each michigan solution
        """
        self.__data_set = data_set

    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        """Run the objective function on the given parameters

        Args:
            solutions (MichiganSolution[]): Solutions that are evaluated
            obj_index (int): Index of the objective in the solution objectives array
            out (double[]): Output array, it will contain the objective value of all the solutions
        """
        cdef int i = 0
        cdef int k = 0
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
                k = winner_rules_indices[i]
                out[k] += 1

            for i in range(len(solutions)):
                sol = solutions[i]
                sol.set_objective(obj_index, out[i])
        else:
            raise InvalidSolutionTypeException("PittsburghSolution")

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "Number of wins"