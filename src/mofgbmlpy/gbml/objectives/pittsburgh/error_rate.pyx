from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution cimport PittsburghSolution

cdef class ErrorRate(ObjectiveFunction):
    """Objective function that uses the error rate as its value

        Attributes:
            __data_set (Dataset): Training dataset
        """
    def __init__(self, data_set):
        """Constructor

        Args:
            data_set (Dataset): Training dataset
        """
        self.__data_set = data_set

    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        """Run the objective function on the given parameters

        Args:
            solutions (PittsburghSolution[]): Solutions that are evaluated
            obj_index (int): Index of the objective in the solution objectives array
            out (double[]): Output array, it will contain the objective value of all the solutions
        """
        cdef int i = 0
        cdef PittsburghSolution sol

        if isinstance(solutions[0], PittsburghSolution):
            for i in range(len(solutions)):
                sol = solutions[i]
                out[i] = sol.get_error_rate(self.__data_set)
                sol.set_objective(obj_index, out[i])
        else:
            raise Exception("Solution must be of type PittsburghSolution")

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "Error rate"