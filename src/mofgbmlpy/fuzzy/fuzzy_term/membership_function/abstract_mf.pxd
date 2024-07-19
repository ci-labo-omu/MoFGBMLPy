import cython
cimport numpy as cnp


cdef class AbstractMF:
    cdef double[:] _params
    cdef bint _are_params_points_flag # If true then we can move them in an interactive plot. e.g. for gaussian it's set to false

    cdef double get_value(self, double x)
    cpdef cnp.ndarray[double, ndim=1] get_params(self)
    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index, double x_min=?, double x_max=?)
    cpdef bint are_params_points(self)
    cpdef bint is_param_value_valid(self, int index, double value, double x_min=?, double x_max=?)
    cpdef void set_param_value(self, int index, double value, double x_min=?, double x_max=?)
    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self, double x_min=?, double x_max=?)