from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution

cdef class TotalRuleLength(ObjectiveFunction):
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out)