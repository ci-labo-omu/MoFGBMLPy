import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set cimport FuzzySet
import cython

cdef class LinguisticVariable:
    cdef double[:] __support_values
    cdef FuzzySet[:] __fuzzy_sets
    cdef str __concept
    cdef double[:] __domain

    cpdef str get_concept(self)
    cdef double get_membership_value(self, int fuzzy_set_index, double x)
    cpdef int get_length(self)
    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index)
    cpdef double get_support(self, int fuzzy_set_id)
    cpdef get_fuzzy_sets(self)
    cpdef get_domain(self)