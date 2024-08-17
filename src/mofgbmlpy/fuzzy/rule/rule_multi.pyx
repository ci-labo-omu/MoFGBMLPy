import copy

from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
import numpy as np
cimport numpy as cnp

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi cimport RuleWeightMulti

cdef class RuleMulti(AbstractRule):
    """Fuzzy rule for multilabel classification """
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
        new_rule = RuleMulti(copy.deepcopy(self.get_antecedent()), copy.deepcopy(self.get_consequent()))
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
        cdef double cf_mean
        cdef RuleWeightMulti rule_weight

        membership = self.get_antecedent().get_compatible_grade_value(attribute_vector)
        rule_weight = self.get_rule_weight()
        cf_mean = rule_weight.get_mean()

        return membership * cf_mean

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"Rule_MultiClass [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"
