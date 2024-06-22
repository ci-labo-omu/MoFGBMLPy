from abc import ABC, abstractmethod

from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent


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

    def set_consequent(self, consequent):
        self._consequent = consequent

    def get_compatible_grade(self, attribute_vector):
        return self._antecedent.get_compatible_grade(attribute_vector)

    def get_compatible_grade_value(self, attribute_vector):
        return self._antecedent.get_compatible_grade_value(attribute_vector)

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
        return self._consequent.get_rule_weight().get_value()

    def set_rule_weight_value(self, rule_weight_value):
        self._consequent.set_rule_weight_value(rule_weight_value)

    def set_class_label_value(self, class_label_value):
        self.get_consequent().set_class_label_value(class_label_value)

    def get_rule_length(self):
        return self.get_antecedent().get_length()

    @abstractmethod
    def get_fitness_value(self, attribute_vector):
        pass

    class RuleBuilderCore:
        _antecedent_factory = None
        _consequent_factory = None
        _knowledge = None

        def __init__(self, antecedent_factory, consequent_factory, knowledge):
            self._antecedent_factory = antecedent_factory
            self._consequent_factory = consequent_factory
            self._knowledge = knowledge

        def create_antecedent(self, num_rules=None):
            return self._antecedent_factory.create(num_rules)

        def create_antecedent_indices(self, num_rules=None, pattern=None):
            if pattern is None:
                return self._antecedent_factory.create_antecedent_indices(num_rules)
            else:
                if not isinstance(self._antecedent_factory, HeuristicAntecedentFactory):
                    raise Exception("The antecedent factory must be HeuristicAntecedentFactory if a pattern is provided")
                if num_rules is not None:
                    print("Warning: num_rules is not considered when a pattern is provided in create_antecedent_indices")
                return self._antecedent_factory.create_antecedent_indices(pattern)

        def create_antecedent_from_indices(self, antecedent_indices):
            return Antecedent(antecedent_indices, self._knowledge)

        def create_consequent(self, antecedent):
            return self._consequent_factory.learning(antecedent)

        def get_knowledge(self):
            return self._knowledge
