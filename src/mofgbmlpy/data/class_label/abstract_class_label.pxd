cimport numpy as cnp
from mofgbmlpy.utility.fused_types cimport int_or_int_array

cdef class AbstractClassLabel:
    cdef cnp.npy_bool __is_rejected

    cpdef object get_class_label_value(self)
    cpdef void set_class_label_value(self, object class_label)
    cpdef cnp.npy_bool is_rejected(self)
    cpdef void set_rejected(self)