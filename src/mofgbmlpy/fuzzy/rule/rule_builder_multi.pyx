import copy

from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.rule_multi cimport RuleMulti

cdef class RuleBuilderMulti(RuleBuilderCore):
    """Rule builder for multilabel classification"""
    def __init__(self, antecedent_factory, consequent_factory, knowledge):
        """Constructor

        Args:
            antecedent_factory (AbstractAntecedentFactory): Antecedent factory
            consequent_factory (AbstractLearning):  Consequent factory
            knowledge (Knowledge): Knowledge base
        """
        super().__init__(antecedent_factory, consequent_factory, knowledge)

    cpdef RuleMulti create(self, Antecedent antecedent):
        """Create a rule from an antecedent
        
        Args:
            antecedent (Antecedent): Antecedent 

        Returns:
            RuleMulti: New rule
        """
        consequent = self._consequent_factory.learning(antecedent)
        return RuleMulti(antecedent, consequent)

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = RuleBuilderMulti(self._antecedent_factory, self._consequent_factory, self._knowledge)

        memo[id(self)] = new_object
        return new_object
