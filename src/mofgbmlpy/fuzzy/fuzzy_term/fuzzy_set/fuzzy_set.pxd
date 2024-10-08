from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF


cdef class FuzzySet:
    cdef AbstractMF __function
    cdef str __term
    cdef int __id
    cdef int __division_type

    cdef double get_membership_value(self, double x)
    cpdef get_term(self)
    cpdef get_function_callable(self)
    cpdef int get_id(self)
    cpdef AbstractMF get_function(self)
    cpdef get_division_type(self)
    cpdef double get_support(self, double x_min=?, double x_max=?)