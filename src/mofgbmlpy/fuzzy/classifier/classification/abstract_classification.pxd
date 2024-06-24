from mofgbmlpy.data.pattern cimport Pattern
cimport numpy as cnp

cdef class AbstractClassification:
    cpdef classify(self, cnp.ndarray[object, ndim=1] michigan_solution_list, Pattern pattern)