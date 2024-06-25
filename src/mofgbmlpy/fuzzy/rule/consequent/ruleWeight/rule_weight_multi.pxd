from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp
import cython


cdef class RuleWeightMulti(AbstractRuleWeight):
    cdef object __rule_weight

    cpdef get_rule_weight_at(self, int index)
    cpdef int get_length(self)
    cpdef object get_value(self)
    cpdef void set_value(self, object rule_weight)
