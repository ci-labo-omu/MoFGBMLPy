from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
import numpy as np
cimport numpy as cnp
import cython

cdef class RuleMulti(AbstractRule):
    cpdef double get_fitness_value(self, cnp.ndarray[double, ndim=1] attribute_vector)