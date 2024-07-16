from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
cimport numpy as cnp

from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.fuzzy.rule.consequent.learning.abstract_learning cimport AbstractLearning

cdef class RuleBuilderCore:
    cdef AbstractAntecedentFactory _antecedent_factory
    cdef AbstractLearning _consequent_factory
    cdef Knowledge _knowledge

    cdef Antecedent create_antecedent(self, int num_rules=?)
    cdef int[:,:] create_antecedent_indices(self, int num_rules=?, Pattern pattern=?)
    cdef Antecedent create_antecedent_from_indices(self, int[:] antecedent_indices)
    cdef Consequent create_consequent(self, Antecedent antecedent)
    cpdef Knowledge get_knowledge(self)
    cpdef Dataset get_training_dataset(self)
