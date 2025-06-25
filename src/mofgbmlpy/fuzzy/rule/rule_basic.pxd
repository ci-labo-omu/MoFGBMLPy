from mofgbmlpy.fuzzy.rule.rule_builder_core import RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
cimport numpy as cnp
import cython


cdef class RuleBasic(AbstractRule):
    cpdef double get_fitness_value(self, double[:] attribute_vector)

