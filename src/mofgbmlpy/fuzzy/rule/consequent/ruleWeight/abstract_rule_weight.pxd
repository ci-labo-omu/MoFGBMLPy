cimport numpy as cnp
import cython

cdef class AbstractRuleWeight:
    cpdef object get_value(self)

    cpdef void set_value(self, object rule_weight)
