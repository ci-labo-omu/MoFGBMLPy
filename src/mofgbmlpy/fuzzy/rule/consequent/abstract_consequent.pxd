import copy
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp


cdef class AbstractConsequent:
    cpdef AbstractClassLabel get_class_label(self)
    cpdef void set_class_label_value(self, object class_label_value)
    cpdef object get_class_label_value(self)
    cpdef bint is_rejected(self)
    cpdef void set_rejected(self)
    cpdef AbstractRuleWeight get_rule_weight(self)
    cpdef void set_rule_weight(self, AbstractRuleWeight rule_weight)
    cpdef str get_linguistic_representation(self)