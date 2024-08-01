import copy

from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
import numpy as np
cimport numpy as cnp


cdef class RuleMulti(AbstractRule):
    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            (object) Deep copy of this object
        """
        new_rule = RuleMulti(copy.deepcopy(self.get_antecedent()), copy.deepcopy(self.get_consequent()))
        memo[id(self)] = new_rule
        return new_rule

    cpdef double get_fitness_value(self, double[:] attribute_vector):
        membership = self.get_antecedent().get_compatible_grade_value(attribute_vector)
        cf_mean = np.mean(self.get_consequent().get_rule_weight_value())
        return membership * cf_mean

    """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"Rule_MultiClass [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"
