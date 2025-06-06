from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
cimport numpy as cnp
import cython

cdef class Antecedent:
    cdef object __antecedent_indices
    cdef Knowledge __knowledge

    cpdef int get_array_size(self)
    cpdef int[:] get_antecedent_indices(self)
    cpdef void set_antecedent_indices(self, int[:] new_indices)
    cpdef float[:] get_membership_values(self, float[:] attribute_vector)
    cdef float get_compatible_grade_value(self, float[:] attribute_vector)
    cpdef int get_length(self)
    cpdef str get_linguistic_representation(self)
    cpdef get_knowledge(self)
    cpdef set_knowledge(self, Knowledge new_knowledge)