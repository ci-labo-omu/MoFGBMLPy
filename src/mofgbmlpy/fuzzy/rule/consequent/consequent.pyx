import copy
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp
from mofgbmlpy.utility.fused_types cimport int_or_int_array
from mofgbmlpy.utility.fused_types cimport double_or_double_array


cdef class Consequent:
    def __init__(self, class_label, rule_weight):
        self._class_label = class_label
        self._rule_weight = rule_weight

    cpdef AbstractClassLabel get_class_label(self):
        return self._class_label

    cpdef void set_class_label_value(self, object class_label_value):
        self._class_label.set_class_label_value(class_label_value)

    cpdef object get_class_label_value(self):
        return self._class_label.get_class_label_value()

    def __eq__(self, other):
        return self.get_class_label_value() == other.get_class_label_value()

    cpdef cnp.npy_bool is_rejected(self):
        return self.get_class_label().is_rejected()

    cpdef void set_rejected(self):
        self._class_label.set_rejected()

    cpdef object get_rule_weight(self):
        return self._rule_weight

    cpdef void set_rule_weight(self, object rule_weight):
        self._rule_weight = rule_weight

    def __copy__(self):
        return Consequent(copy.copy(self._class_label), copy.copy(self._rule_weight))

    def __str__(self):
        return f"class:[{self._class_label}]: weight:[{self._rule_weight}]"
