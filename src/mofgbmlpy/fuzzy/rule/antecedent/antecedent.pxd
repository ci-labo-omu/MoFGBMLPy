from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
cimport numpy as cnp

cdef class Antecedent:
    cdef public object __antecedent_indices
    cdef public Knowledge __knowledge

    cpdef int get_array_size(self)
    cpdef cnp.ndarray[int, ndim=1] get_antecedent_indices(self)
    cpdef void set_antecedent_indices(self, cnp.ndarray[int, ndim=1] new_indices)
    cpdef cnp.ndarray[double, ndim=1] get_compatible_grade(self, cnp.ndarray[double, ndim=1] attribute_vector)
    cdef double get_compatible_grade_value(self, cnp.ndarray[double, ndim=1] attribute_vector)
    cpdef int get_length(self)
