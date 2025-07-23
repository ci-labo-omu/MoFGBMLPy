import math

from matplotlib import pyplot as plt
cimport numpy as cnp
import cython
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable

cdef class Knowledge:
    cdef FuzzyVariable[:] __fuzzy_vars

    cpdef FuzzyVariable get_fuzzy_variable(self, int dim)
    cpdef FuzzySet get_fuzzy_set(self, int dim, int fuzzy_set_id)
    cpdef int get_num_fuzzy_sets(self, int dim)
    cpdef void set_fuzzy_vars(self, FuzzyVariable[:] fuzzy_vars)
    cpdef FuzzyVariable[:] get_fuzzy_vars(self)
    cdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_index)
    cpdef double get_membership_value_py(self, double attribute_value, int dim, int fuzzy_set_index)
    cpdef int get_num_dim(self)
    cpdef double get_support(self, int dim, int fuzzy_set_index)