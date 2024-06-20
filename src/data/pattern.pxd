cdef class Pattern:
    cdef int __id
    cdef object __attribute_vector
    cdef object __target_class

    cpdef int get_id(self)
    cpdef object get_attributes_vector(self)
    cpdef double get_attribute_value(self, index)
    cpdef object get_target_class(self)
