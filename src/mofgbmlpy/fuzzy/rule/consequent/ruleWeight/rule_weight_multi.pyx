from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp


cdef class RuleWeightMulti(AbstractRuleWeight):
    def __init__(self, cnp.ndarray[double, ndim=1] rule_weight):
        self.__rule_weight = rule_weight

    def __copy__(self):
        return RuleWeightMulti(self.get_value())

    def __str__(self):
        if self.get_value() is None:
            raise ValueError("Rule weight is None")

        txt = f"{self.get_value()[0]:.4f}"

        if self.get_length() > 1:
            for i in range(1, self.get_length()):
                txt = f"{txt}, {self.get_value()[i]:.4f}"

        return txt

    cpdef get_rule_weight_at(self, int index):
        return self.get_value()[index]

    cpdef int get_length(self):
        return self.get_value().size

    cpdef object get_value(self):
        return self.__rule_weight

    cpdef void set_value(self, object rule_weight):
        self.__rule_weight = rule_weight