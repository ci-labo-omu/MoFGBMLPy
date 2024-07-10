from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
import cython
cimport numpy as cnp


cdef class RectangularMF(AbstractMF):
    cdef double get_value(self, double x)
    cpdef cnp.ndarray[double, ndim=1] get_params(self)
    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index)
    cpdef bint is_param_value_valid(self, int index, double value)
    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self)