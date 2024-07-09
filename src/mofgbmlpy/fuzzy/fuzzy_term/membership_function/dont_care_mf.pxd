from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
import cython


cdef class DontCareMF(AbstractMF):
    cdef double get_value(self, double _)
    cpdef double[:,:] get_plot_points(self)