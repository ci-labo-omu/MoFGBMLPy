import cython

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF

cdef class FuzzySet:
    cdef AbstractMF __function
    cdef str __term

    cdef double get_membership_value(self, double x)
    cpdef get_term(self)
    cpdef get_function_callable(self)
