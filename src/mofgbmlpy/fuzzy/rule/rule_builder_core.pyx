from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
cimport numpy as cnp

from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent

cdef class RuleBuilderCore:
    def __init__(self, antecedent_factory, consequent_factory, knowledge):
        self._antecedent_factory = antecedent_factory
        self._consequent_factory = consequent_factory
        self._knowledge = knowledge

    cdef Antecedent create_antecedent(self, int num_rules=1):
        return self._antecedent_factory.create(num_rules)

    cdef cnp.ndarray[int, ndim=2] create_antecedent_indices(self, int num_rules=1, Pattern pattern=None):
        if pattern is None:
            return self._antecedent_factory.create_antecedent_indices(num_rules)
        else:
            if not isinstance(self._antecedent_factory, HeuristicAntecedentFactory):
                raise Exception("The antecedent factory must be HeuristicAntecedentFactory if a pattern is provided")
            if num_rules is not None:
                print("Warning: num_rules is not considered when a pattern is provided in create_antecedent_indices")
            return self._antecedent_factory.create_antecedent_indices(pattern)

    cdef Antecedent create_antecedent_from_indices(self, cnp.ndarray[int, ndim=1] antecedent_indices):
        return Antecedent(antecedent_indices, self._knowledge)

    cdef Consequent create_consequent(self, Antecedent antecedent):
        return self._consequent_factory.learning(antecedent)

    cpdef Knowledge get_knowledge(self):
        return self._knowledge