from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF


cdef class FuzzySet:
    cdef AbstractMF __function
    cdef str __term
    cdef int __id
    cdef int __division_type

    cdef float get_membership_value(self, float x)
    cpdef get_term(self)
    cpdef get_function_callable(self)
    cpdef int get_id(self)
    cpdef AbstractMF get_function(self)
    cpdef set_function(self, AbstractMF function)
    cpdef get_division_type(self)
    cpdef float get_support(self, float x_min=?, float x_max=?)