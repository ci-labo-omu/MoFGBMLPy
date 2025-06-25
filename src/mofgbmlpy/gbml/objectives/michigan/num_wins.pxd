import numpy as np
from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class NumWins(ObjectiveFunction):
    cdef Dataset __data_set
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out)