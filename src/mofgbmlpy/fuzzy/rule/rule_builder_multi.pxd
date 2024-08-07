from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.rule_multi cimport RuleMulti

cdef class RuleBuilderMulti(RuleBuilderCore):
    cpdef RuleMulti create(self, Antecedent antecedent)