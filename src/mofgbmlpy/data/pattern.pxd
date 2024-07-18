cimport numpy as cnp
import cython
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel

cdef class Pattern:
    cdef int __id
    cdef double[:] __attributes_vector
    cdef AbstractClassLabel __target_class

    cpdef int  get_id(self)
    cpdef double[:] get_attributes_vector(self)
    cpdef double get_attribute_value(self, int index)
    cpdef object get_target_class(self)
    cpdef int get_num_dim(self)
