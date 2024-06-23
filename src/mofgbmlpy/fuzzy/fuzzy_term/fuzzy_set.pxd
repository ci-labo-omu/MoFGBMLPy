cdef class FuzzySet:
    cdef object __function
    cdef str __term

    cpdef double get_membership_value(self, double x)
