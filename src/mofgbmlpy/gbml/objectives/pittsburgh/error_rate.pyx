from mofgbmlpy.gbml.objectives.ObjectiveFunction cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution cimport PittsburghSolution

cdef class ErrorRate(ObjectiveFunction):
    def __init__(self, data_set):
        self.__data_set = data_set

    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        cdef int i = 0
        cdef PittsburghSolution sol

        if isinstance(solutions[0], PittsburghSolution):
            for i in range(len(solutions)):
                sol = solutions[i]
                out[i] = sol.get_error_rate(self.__data_set)
                sol.set_objective(obj_index, out[i])
        else:
            raise Exception("Solution must be of type PittsburghSolution")