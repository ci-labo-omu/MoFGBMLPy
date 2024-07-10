from mofgbmlpy.gbml.objectives.ObjectiveFunction cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution

cdef class NumRules(ObjectiveFunction):
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out)