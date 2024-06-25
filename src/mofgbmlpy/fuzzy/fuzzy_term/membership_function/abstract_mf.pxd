import cython

cdef class AbstractMF:
    cpdef double get_value(self, double x)
