from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
import cython
cimport numpy as cnp

cdef class TriangularMF(AbstractMF):
    cdef double get_value(self, double x)
    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index, double x_min=?, double x_max=?)
    cpdef bint is_param_value_valid(self, int index, double value, double x_min=?, double x_max=?)
    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self, double x_min=?, double x_max=?)