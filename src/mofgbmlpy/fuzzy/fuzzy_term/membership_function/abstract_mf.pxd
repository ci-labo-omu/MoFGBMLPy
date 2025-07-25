import cython
cimport numpy as cnp

from libcpp.vector cimport vector
from libcpp.string cimport string as std_string


cdef extern from "core/fuzzy/fuzzy_term/membership_function/abstract_mf.hpp":
    cdef cppclass AbstractMFCpp "AbstractMF":
        AbstractMFCpp(const vector[float]& params, bint are_params_points_flag);
        float get_value(float x) const;
        vector[float] get_params() const;
        vector[float] get_param_range(int index, float x_min, float x_max) except +;
        bint are_params_points() const;
        bint is_param_value_valid(int index, float value, float x_min, float x_max) except +;
        void set_param_value(int index, float value, float x_min, float x_max) except +;
        vector[vector[float]] get_plot_points(float x_min, float x_max) const
        float get_support(float x_min, float x_max) const
        bint operator==(const AbstractMFCpp& other) const;
        std_string to_string() const;
        AbstractMFCpp* clone() const

cdef class AbstractMF:
    cdef AbstractMFCpp* ptr

    cdef float get_value(self, float x)
    cpdef cnp.ndarray[float, ndim=1] get_params(self)
    cpdef cnp.ndarray[float, ndim=1] get_param_range(self, int index, float x_min=?, float x_max=?)
    cpdef bint are_params_points(self)
    cpdef bint is_param_value_valid(self, int index, float value, float x_min=?, float x_max=?)
    cpdef void set_param_value(self, int index, float value, float x_min=?, float x_max=?)
    cpdef cnp.ndarray[float, ndim=2] get_plot_points(self, float x_min=?, float x_max=?)
    cpdef float get_support(self, float x_min=?, float x_max=?)
