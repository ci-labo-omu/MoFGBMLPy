import copy

from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.rule_basic cimport RuleBasic


cdef class RuleBuilderBasic(RuleBuilderCore):
    """Rule builder for basic classification (no multilabel)"""
    def __init__(self, antecedent_factory, consequent_factory, knowledge):
        """Constructor

        Args:
            antecedent_factory (AbstractAntecedentFactory): Antecedent factory
            consequent_factory (AbstractLearning):  Consequent factory
            knowledge (Knowledge): Knowledge base
        """
        super().__init__(antecedent_factory, consequent_factory, knowledge)

    cpdef RuleBasic create(self, Antecedent antecedent):
        """Create a rule from an antecedent

        Args:
            antecedent (Antecedent): Antecedent 

        Returns:
            RuleBasic: New rule
        """
        consequent = self._consequent_factory.learning(antecedent)
        return RuleBasic(antecedent, consequent)

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = RuleBuilderBasic(self._antecedent_factory, self._consequent_factory, self._knowledge)

        memo[id(self)] = new_object
        return new_object
