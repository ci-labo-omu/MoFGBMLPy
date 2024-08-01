cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern
import cython
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class AbstractClassification:
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern):
        raise Exception("AbstractClassification is abstract")

    def __copy__(self):
        raise Exception("AbstractClassification is abstract")
