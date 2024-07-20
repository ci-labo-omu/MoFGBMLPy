import copy

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
import numpy as np


cdef class AllCombinationAntecedentFactory(AbstractAntecedentFactory):
    cdef int[:,:] __antecedents_indices
    cdef Knowledge __knowledge

    cdef int[:,:] generate_antecedents_indices(self)
    cdef Antecedent[:] create(self, int num_rules=?)
    cdef int[:,:] create_antecedent_indices(self, int num_rules=?)
    cpdef Knowledge get_knowledge(self)
