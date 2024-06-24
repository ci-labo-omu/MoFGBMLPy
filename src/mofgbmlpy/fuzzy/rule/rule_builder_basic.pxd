from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.rule_basic cimport RuleBasic


cdef class RuleBuilderBasic(RuleBuilderCore):
    cpdef RuleBasic create(self, Antecedent antecedent)