from fuzzy.rule.abstract_rule import AbstractRule
from fuzzy.rule.antecedent.antecedent import Antecedent


class RuleBasic(AbstractRule):
    def __init__(self, antecedent, consequent):
        super().__init__(antecedent, consequent)

    def __copy__(self):
        return RuleBasic(self.get_antecedent(), self.get_consequent())

    def get_fitness_value(self, attribute_vector):
        membership = self.get_antecedent().get_compatible_grade_value(attribute_vector)
        cf = self.get_rule_weight().get_value()
        return membership * cf

    def __str__(self):
        return f"Rule_Basic [antecedent={self.get_antecedent()}, consequent={self.get_consequent()}]"

    class RuleBuilderBasic(AbstractRule.RuleBuilderCore):
        def __init__(self, antecedent_factory, consequent_factory, knowledge):
            super().__init__(antecedent_factory, consequent_factory, knowledge)

        def create(self, antecedent):
            consequent = self._consequent_factory.learning(antecedent)
            return RuleBasic(antecedent, consequent)

        def __copy__(self):
            return RuleBasic.RuleBuilderBasic(self._antecedent_factory, self._consequent_factory, self._knowledge)
