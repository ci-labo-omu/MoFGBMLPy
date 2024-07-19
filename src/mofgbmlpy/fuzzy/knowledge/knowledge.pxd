import math

from matplotlib import pyplot as plt
cimport numpy as cnp
import cython
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable

cdef class Knowledge:
    cdef public FuzzyVariable[:] __fuzzy_sets

    cpdef FuzzyVariable get_fuzzy_variable(self, int dim)
    cpdef FuzzySet get_fuzzy_set(self, int dim, int fuzzy_set_id)
    cpdef int get_num_fuzzy_sets(self, int dim)
    cpdef void set_fuzzy_sets(self, FuzzyVariable[:] fuzzy_sets)
    cpdef FuzzyVariable[:] get_fuzzy_sets(self)
    cdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_id)
    cpdef double get_membership_value_py(self, double attribute_value, int dim, int fuzzy_set_id)
    cpdef int get_num_dim(self)
    cpdef void clear(self)
    cpdef double get_support(self, int dim, int fuzzy_set_id)