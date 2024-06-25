cimport numpy as cnp

from mofgbmlpy.utility.fused_types cimport double_or_double_array

cdef class AbstractRuleWeight:
    cpdef object get_value(self):
        # with cython.gil:
        raise Exception("AbstractRuleWeight is abstract")

    cpdef void set_value(self, object rule_weight):
        # with cython.gil:
        raise Exception("AbstractRuleWeight is abstract")
