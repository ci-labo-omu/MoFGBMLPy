from src.fuzzy.rule.abstract_rule import AbstractRule
import numpy as np


class RuleMulti(AbstractRule):
    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)

    def copy(self):
        return RuleMulti(self.get_antecedent(), self.get_consequent())

    def get_fitness_value(self, attribute_vector):
        membership = self.get_antecedent().get_compatible_grade_value(attribute_vector)
        cf_mean = np.mean(self.get_consequent().get_rule_weight_value())
        return membership * cf_mean

    def get_rule_length(self):
        return self.get_antecedent().get_rule_length()

    def __str__(self):
        return f"Rule_MultiClass [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"
