import math

from matplotlib import pyplot as plt
cimport numpy as cnp
from simpful import LinguisticVariable

cdef class Knowledge:
    cdef public object __fuzzy_sets

    cpdef object get_fuzzy_variable(self, int dim)
    cpdef object get_fuzzy_set(self, int dim, int fuzzy_set_id)
    cpdef int get_num_fuzzy_sets(self, int dim)
    cpdef void set_fuzzy_sets(self, cnp.ndarray[object, ndim=1] fuzzy_sets)
    cpdef cnp.ndarray[object, ndim=1] get_fuzzy_sets(self)
    cpdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_id)
    cpdef int get_num_dim(self)
    cpdef void clear(self)
    cpdef float get_support(self, int dim, int fuzzy_set_id)