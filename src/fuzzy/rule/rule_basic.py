from src.fuzzy.rule.abstract_rule import AbstractRule


class RuleBasic(AbstractRule):
    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)

    def copy(self):
        return RuleBasic(self.get_antecedent(), self.get_consequent())

    def get_fitness_value(self, antecedent_indices, attribute_vector):
        membership = self.get_antecedent().get_compatible_grade_value(antecedent_indices, attribute_vector)
        cf = self.get_rule_weight().get_rule_weight_value()
        return membership * cf

    def set_class_label_value(self, class_label_value):
        self.get_consequent().set_class_label_value(class_label_value)

    def __str__(self):
        return f"Rule_Basic [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"
