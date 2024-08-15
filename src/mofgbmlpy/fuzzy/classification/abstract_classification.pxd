from mofgbmlpy.data.pattern cimport Pattern
cimport numpy as cnp
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class AbstractClassification:
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern)