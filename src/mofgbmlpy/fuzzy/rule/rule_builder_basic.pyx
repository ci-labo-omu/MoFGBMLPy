from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.rule_basic cimport RuleBasic


cdef class RuleBuilderBasic(RuleBuilderCore):
    def __init__(self, antecedent_factory, consequent_factory, knowledge):
        super().__init__(antecedent_factory, consequent_factory, knowledge)

    cpdef RuleBasic create(self, Antecedent antecedent):
        consequent = self._consequent_factory.learning(antecedent)
        return RuleBasic(antecedent, consequent)

    def __copy__(self):
        return RuleBuilderBasic(self._antecedent_factory, self._consequent_factory, self._knowledge)