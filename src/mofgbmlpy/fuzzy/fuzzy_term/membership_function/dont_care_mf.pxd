from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF, AbstractMFCpp
cimport numpy as cnp

from libcpp.vector cimport vector


cdef extern from "core/fuzzy/fuzzy_term/membership_function/dont_care_mf.hpp":
    cdef cppclass DontCareMFCpp "DontCareMF"(AbstractMFCpp):
        DontCareMF();
    

cdef class DontCareMF(AbstractMF):
    cdef float get_value(self, float _)
    cpdef cnp.ndarray[float, ndim=1] get_param_range(self, int index, float x_min=?, float x_max=?)
    cpdef bint is_param_value_valid(self, int index, float value, float x_min=?, float x_max=?)
    cpdef cnp.ndarray[float, ndim=2] get_plot_points(self, float x_min=?, float x_max=?)
    cpdef float get_support(self, float x_min=?, float x_max=?)
    @staticmethod
    cdef DontCareMF wrap(DontCareMFCpp * wrapped_ptr)
