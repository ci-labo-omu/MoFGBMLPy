from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.data.dataset_density cimport DatasetWithDensity
from mofgbmlpy.data.dataset_manager cimport DatasetManager

cdef class ErrorRate(ObjectiveFunction):
    #cdef Dataset __data_set
    cdef DatasetWithDensity __data_set
    cdef DatasetManager __dataset_manager
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out)