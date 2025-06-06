cimport numpy as cnp
import cython
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel

cdef class Pattern:
    cdef int __id
    cdef float[:] __attributes_vector
    cdef AbstractClassLabel __target_class

    cpdef int  get_id(self)
    cpdef float[:] get_attributes_vector(self)
    cpdef float get_attribute_value(self, int index)
    cpdef object get_target_class(self)
    cpdef int get_num_dim(self)
