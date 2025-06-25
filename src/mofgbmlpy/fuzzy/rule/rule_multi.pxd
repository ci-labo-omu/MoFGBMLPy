from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
import numpy as np
cimport numpy as cnp
import cython

cdef class RuleMulti(AbstractRule):
    cpdef double get_fitness_value(self, double[:] attribute_vector)