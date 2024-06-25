from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight


cdef class RuleWeightBasic(AbstractRuleWeight):
    def __init__(self, double rule_weight):
        self.__rule_weight = rule_weight

    def __copy__(self):
        return RuleWeightBasic(self.get_value())

    def __str__(self):
        if self.get_value() is None:
            # with cython.gil:
            raise ValueError("Rule weight is None")
        return f"{self.get_value():.4f}"

    cpdef object get_value(self):
        return self.__rule_weight

    cpdef void set_value(self, object rule_weight):
        self.__rule_weight = rule_weight

    def __eq__(self, other):
        return self.__rule_weight == other.get_value()