from abc import ABC


class AbstractRule(ABC):
    _antecedent = None
    _consequent = None

    def __init__(self, antecedent, consequent):
        self._antecedent = antecedent
        self._consequent = consequent

    def get_antecedent(self):
        return self._antecedent

    def get_consequent(self):
        return self._consequent

    def get_compatible_grade(self, antecedent_index, attribute_vector):
        return self._antecedent.get_compatible_grade(antecedent_index, attribute_vector)

    def get_compatible_grade_value(self, antecedent_index, attribute_vector):
        return self._antecedent.get_compatible_grade_value(antecedent_index, attribute_vector)

    def get_class_label(self):
        return self._consequent.get_class_label()

    def get_class_label_value(self):
        return self._consequent.get_class_label_value()

    def equals_class_label(self, other):
        return self._consequent().get_class_label() == other.get_consequent().get_class_label()

    def is_rejected_class_label(self):
        return self._consequent.get_class_label().is_rejected()

    def get_rule_weight(self):
        return self._consequent.get_rule_weight()

    def get_rule_weight_value(self):
        return self._consequent.get_rule_weight_value()

    def set_rule_weight_value(self, rule_weight_value):
        self._consequent.set_rule_weight_value(rule_weight_value)

    # TODO: Add rule builder ?
