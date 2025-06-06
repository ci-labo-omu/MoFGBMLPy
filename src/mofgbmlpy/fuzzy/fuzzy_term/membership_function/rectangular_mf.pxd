from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
import cython
cimport numpy as cnp


cdef class RectangularMF(AbstractMF):
    cdef float get_value(self, float x)
    cpdef cnp.ndarray[float, ndim=1] get_param_range(self, int index, float x_min=?, float x_max=?)
    cpdef bint is_param_value_valid(self, int index, float value, float x_min=?, float x_max=?)
    cpdef cnp.ndarray[float, ndim=2] get_plot_points(self, float x_min=?, float x_max=?)
    cpdef float get_support(self, float x_min=?, float x_max=?)