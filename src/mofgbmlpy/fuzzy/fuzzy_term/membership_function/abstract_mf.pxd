import cython
cimport numpy as cnp


cdef class AbstractMF:
    cdef float[:] _params
    cdef bint _are_params_points_flag # If true then we can move them in an interactive plot. e.g. for gaussian it's set to false

    cdef float get_value(self, float x)
    cpdef cnp.ndarray[float, ndim=1] get_params(self)
    cpdef cnp.ndarray[float, ndim=1] get_param_range(self, int index, float x_min=?, float x_max=?)
    cpdef bint are_params_points(self)
    cpdef bint is_param_value_valid(self, int index, float value, float x_min=?, float x_max=?)
    cpdef void set_param_value(self, int index, float value, float x_min=?, float x_max=?)
    cpdef cnp.ndarray[float, ndim=2] get_plot_points(self, float x_min=?, float x_max=?)
    cpdef float get_support(self, float x_min=?, float x_max=?)