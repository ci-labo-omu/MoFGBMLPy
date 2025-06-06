from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp
import cython


cdef class RuleWeightMulti(AbstractRuleWeight):
    cdef float[:] __rule_weight

    cpdef get_rule_weight_at(self, int index)
    cpdef int get_length(self)
    cpdef object get_value(self)
    cpdef void set_value(self, object rule_weight)
    cdef float get_mean(self)
    cpdef float get_mean_py(self)