from mofgbmlpy.data.dataset_density cimport DatasetWithDensity
from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution

cdef class ErrorRate(ObjectiveFunction):
    cdef DatasetWithDensity __data_set
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out)