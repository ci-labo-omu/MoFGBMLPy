from mofgbmlpy.fuzzy.rule.rule_builder_core import RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
cimport numpy as cnp
import cython


cdef class RuleBasic(AbstractRule):
    cpdef double get_fitness_value(self, cnp.ndarray[double, ndim=1] attribute_vector)

