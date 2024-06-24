from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
cimport numpy as cnp

from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent

cdef class RuleBuilderCore:
    cdef object _antecedent_factory
    cdef object _consequent_factory
    cdef object _knowledge

    cdef Antecedent create_antecedent(self, int num_rules=?)
    cdef cnp.ndarray[int, ndim=2] create_antecedent_indices(self, int num_rules=?, Pattern pattern=?)
    cdef Antecedent create_antecedent_from_indices(self, cnp.ndarray[int, ndim=1] antecedent_indices)
    cdef Consequent create_consequent(self, Antecedent antecedent)
    cpdef Knowledge get_knowledge(self)
