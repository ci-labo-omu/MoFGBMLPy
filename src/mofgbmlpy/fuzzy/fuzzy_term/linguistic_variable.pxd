import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set cimport FuzzySet
import cython

cdef class LinguisticVariable:
    cdef object __support_values
    cdef object __fuzzy_sets
    cdef str __concept
    cdef object __domain

    cpdef void add_fuzzy_set(self, FuzzySet fuzzy_set, double support_value)
    cpdef str get_concept(self)
    cpdef double get_membership_value(self, int fuzzy_set_index, double x)
    cpdef int get_length(self)
    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index)
    cpdef double get_support(self, int fuzzy_set_id)
