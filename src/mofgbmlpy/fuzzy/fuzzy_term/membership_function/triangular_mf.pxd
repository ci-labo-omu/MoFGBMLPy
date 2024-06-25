from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
import cython

cdef class TriangularMF(AbstractMF):
    cdef double __left
    cdef double __center
    cdef double __right

    cpdef double get_value(self, double x)