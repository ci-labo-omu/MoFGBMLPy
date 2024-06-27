from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
cimport numpy as cnp
import cython

cdef class Antecedent:
    cdef public object __antecedent_indices
    cdef public Knowledge __knowledge

    cpdef int get_array_size(self)
    cpdef int[:] get_antecedent_indices(self)
    cpdef void set_antecedent_indices(self, int[:] new_indices)
    cpdef cnp.ndarray[double, ndim=1] get_compatible_grade(self, cnp.ndarray[double, ndim=1] attribute_vector)
    cdef double get_compatible_grade_value(self, cnp.ndarray[double, ndim=1] attribute_vector)
    cpdef double get_compatible_grade_value_py(self, cnp.ndarray[double, ndim=1] attribute_vector)
    cpdef int get_length(self)
