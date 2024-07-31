import copy

from mofgbmlpy.fuzzy.rule.rule_builder_core import RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
cimport numpy as cnp


cdef class RuleBasic(AbstractRule):
    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            Deep copy of this object
        """
        new_rule = RuleBasic(copy.deepcopy(self.get_antecedent()), copy.deepcopy(self.get_consequent()))
        memo[id(self)] = new_rule
        return new_rule

    cpdef double get_fitness_value(self, double[:] attribute_vector):
        cdef double membership
        cdef double cf
        membership = self.get_antecedent().get_compatible_grade_value(attribute_vector)
        cf = self.get_rule_weight().get_value()
        return membership * cf

    def __str__(self):
        return f"Rule_Basic [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"

