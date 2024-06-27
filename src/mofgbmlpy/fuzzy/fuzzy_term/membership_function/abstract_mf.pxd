import cython

cdef class AbstractMF:
    cdef double get_value(self, double x)
