import xml.etree.cElementTree as xml_tree
cimport numpy as cnp
import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF

cdef class DontCareMF(AbstractMF):
    def __init__(self):
        super().__init__(None)

    cdef double get_value(self, double _):
        return 1.0

    def __str__(self):
        return "<Dont Care MF>"

    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index, double x_min=0, double x_max=1):
        return np.empty(0, dtype=np.float64)

    cpdef bint is_param_value_valid(self, int index, double value, double x_min=0, double x_max=1):
        return False # Can't be edited

    def __deepcopy__(self, memo={}):
        new_object = DontCareMF()
        memo[id(self)] = new_object
        return new_object

    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self, double x_min=0, double x_max=1):
        return np.array([[0,1], [1,1]])