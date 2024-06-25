from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
import cython


cdef class DontCareMF(AbstractMF):
    cpdef double get_value(self, double _)