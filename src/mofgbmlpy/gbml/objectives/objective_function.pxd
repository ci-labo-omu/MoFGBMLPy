from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution

cdef class ObjectiveFunction:
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out)