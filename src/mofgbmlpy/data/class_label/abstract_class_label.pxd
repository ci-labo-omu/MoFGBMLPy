cimport numpy as cnp

cdef class AbstractClassLabel:
    cdef bint __is_rejected

    cpdef object get_class_label_value(self)
    cpdef void set_class_label_value(self, object class_label)
    cpdef bint is_rejected(self)
    cpdef void set_rejected(self)