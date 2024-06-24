from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
import numpy as np
cimport numpy as cnp


cdef class RuleMulti(AbstractRule):
    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)

    def __copy__(self):
        return RuleMulti(self.get_antecedent(), self.get_consequent())

    cdef double get_fitness_value(self, cnp.ndarray[double, ndim=1] attribute_vector):
        membership = self.get_antecedent().get_compatible_grade_value(attribute_vector)
        cf_mean = np.mean(self.get_consequent().get_rule_weight_value())
        return membership * cf_mean

    def __str__(self):
        return f"Rule_MultiClass [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"
