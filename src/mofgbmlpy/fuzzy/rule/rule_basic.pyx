import copy

from mofgbmlpy.fuzzy.rule.rule_builder_core import RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
cimport numpy as cnp


cdef class RuleBasic(AbstractRule):
    """Fuzzy rule for simple classification (not multilabel) """

    def __init__(self, antecedent, consequent):
        """Constructor

        Args:
            antecedent (Antecedent): Antecedent of the rule
            consequent (Consequent): Consequent of the rule
        """
        super().__init__(antecedent, consequent)

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_rule = RuleBasic(copy.deepcopy(self.get_antecedent()), copy.deepcopy(self.get_consequent()))
        memo[id(self)] = new_rule
        return new_rule

    cpdef double get_fitness_value(self, double[:] attribute_vector):
        """Get the fitness value of the rule for the given input vector

        Args:
            attribute_vector (double[]): Input vector 

        Returns:
            double: Fitness value
        """
        cdef double membership
        cdef double cf
        membership = self.get_antecedent().get_compatible_grade_value(attribute_vector)
        cf = self.get_rule_weight().get_value()
        return membership * cf

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"Rule_Basic [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"

