from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent



cdef class HeuristicAntecedentFactory(AbstractAntecedentFactory):
    cdef Dataset __training_set
    cdef Knowledge __knowledge
    cdef bint __is_dc_probability
    cdef double __dc_rate
    cdef int __antecedent_number_do_not_dont_care
    cdef object _random_gen

    cdef int[:] __select_antecedent_part(self, int index)
    cdef int[:] calculate_antecedent_part(self, Pattern pattern)
    cdef Antecedent[:] create(self, int num_rules=?)
    cdef int[:,:] create_antecedent_indices_from_pattern(self, Pattern pattern=?)
    cdef int[:,:] create_antecedent_indices(self, int num_rules=?)