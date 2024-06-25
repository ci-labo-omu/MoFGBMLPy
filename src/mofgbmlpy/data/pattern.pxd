cimport numpy as cnp
import cython

cdef class Pattern:
    cdef int __id
    cdef object __attribute_vector
    cdef object __target_class

    cpdef int  get_id(self)
    cpdef cnp.ndarray[object, ndim=1] get_attributes_vector(self)
    cpdef double get_attribute_value(self, int index)
    cpdef object get_target_class(self)
