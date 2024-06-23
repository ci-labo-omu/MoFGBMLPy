from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF

cdef class DontCareMF(AbstractMF):
    cpdef double get_value(self, double _):
        return 1.0

    def __str__(self):
        return "<Dont Care MF>"