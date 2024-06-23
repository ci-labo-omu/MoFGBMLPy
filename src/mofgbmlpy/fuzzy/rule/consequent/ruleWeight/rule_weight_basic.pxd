from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight



cdef class RuleWeightBasic(AbstractRuleWeight):
    cdef double __rule_weight

    cpdef object get_value(self)
    cpdef void set_value(self, object rule_weight)
