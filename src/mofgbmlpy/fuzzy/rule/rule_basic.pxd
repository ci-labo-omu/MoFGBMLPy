from mofgbmlpy.fuzzy.rule.rule_builder_core import RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
cimport numpy as cnp
import cython


cdef class RuleBasic(AbstractRule):
    cpdef float get_fitness_value(self, float[:] attribute_vector)

