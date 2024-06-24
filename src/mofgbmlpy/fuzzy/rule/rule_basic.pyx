from mofgbmlpy.fuzzy.rule.rule_builder_core import RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
cimport numpy as cnp


cdef class RuleBasic(AbstractRule):
    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)

    def __copy__(self):
        return RuleBasic(self.get_antecedent(), self.get_consequent())

    cpdef double get_fitness_value(self, cnp.ndarray[double, ndim=1] attribute_vector):
        cdef double membership
        cdef double cf
        membership = self.get_antecedent().get_compatible_grade_value(attribute_vector)
        cf = self.get_rule_weight().get_value()
        return membership * cf

    def __str__(self):
        return f"Rule_Basic [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"

