from src.fuzzy.rule.abstract_rule import AbstractRule
import numpy as np


class RuleMulti(AbstractRule):
    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)

    def copy(self):
        return RuleMulti(self.get_antecedent(), self.get_consequent())

    def get_fitness_value(self, antecedent_indices, attribute_vector):
        membership = self.get_antecedent().get_compatible_grade_value(antecedent_indices, attribute_vector)
        cf_mean = np.mean(self.get_consequent().get_rule_weight_value())
        return membership * cf_mean

    def set_class_label_value(self, class_label_value):
        self.get_consequent().set_class_label_value(class_label_value)

    def get_rule_length(self, antecedent_indices):
        return self.get_antecedent().get_rule_length(antecedent_indices)

    def __str__(self):
        return f"Rule_MultiClass [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"
