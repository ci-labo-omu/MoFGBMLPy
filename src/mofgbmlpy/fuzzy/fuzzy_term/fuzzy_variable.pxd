import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
import cython

cdef class FuzzyVariable:
    cdef FuzzySet[:] __fuzzy_sets
    cdef str __name
    cdef double[:] __domain

    cpdef str get_name(self)
    cdef double get_membership_value(self, int fuzzy_set_index, double x)
    cpdef int get_length(self)
    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index)
    cpdef double get_support(self, int fuzzy_set_index)
    cpdef get_fuzzy_sets(self)
    cpdef get_support_values(self)
    cpdef get_domain(self)