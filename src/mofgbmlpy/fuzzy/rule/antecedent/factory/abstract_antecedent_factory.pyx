from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent


cdef class AbstractAntecedentFactory:
    cdef Antecedent[:] create(self, int num_rules=1):
        pass

    cdef int[:,:] create_antecedent_indices(self, int num_rules=1):
        pass

    def __copy__(self):
        pass
