from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent

cdef class AbstractAntecedentFactory:
    cdef Antecedent[:] create(self, int num_rules=?)
    cdef int[:,:] create_antecedent_indices(self, int num_rules=?)
