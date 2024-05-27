from abc import abstractmethod, ABC


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

