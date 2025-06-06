from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
import numpy as np
cimport numpy as cnp
import cython

cdef class RuleMulti(AbstractRule):
    cpdef float get_fitness_value(self, float[:] attribute_vector)