cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern


cdef class AbstractClassification:
    cpdef classify(self, cnp.ndarray[object, ndim=1] michigan_solution_list, Pattern pattern):
        raise Exception("AbstractClassification is abstract")

    def __copy__(self):
        raise Exception("AbstractClassification is abstract")
