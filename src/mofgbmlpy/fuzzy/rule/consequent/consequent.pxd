import copy
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp


cdef class Consequent:
    cdef AbstractClassLabel _class_label
    cdef AbstractRuleWeight _rule_weight

    cpdef AbstractClassLabel get_class_label(self)
    cpdef void set_class_label_value(self, object class_label_value)
    cpdef object get_class_label_value(self)
    cpdef cnp.npy_bool is_rejected(self)
    cpdef void set_rejected(self)
    cpdef object get_rule_weight(self)
    cpdef void set_rule_weight(self, object rule_weight)
