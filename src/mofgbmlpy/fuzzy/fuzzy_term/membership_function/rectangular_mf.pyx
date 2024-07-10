import xml.etree.cElementTree as xml_tree

import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF

cdef class RectangularMF(AbstractMF):
    def __init__(self, left, right):
        if left >= right:
            raise Exception(f"Error in triangular membership function: left={left:.2f} should be <= right={right:.2f}")
        super().__init__(np.array([left, right]))

    cdef double get_value(self, double x):
        return 1 if x >= self._params[0] and x <= self._params[1] else 0

    def __str__(self):
        return "<Rectangular MF>"

    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index):
        if index == 0:
            return np.array([0, self._params[1]])
        elif index == 1:
            return np.array([self._params[0], 1])
        else:
            raise Exception("Invalid index for rectangular MF")

    cpdef bint is_param_value_valid(self, int index, double value):
        if index == 0:
            if value > self._params[1]:
                return False
        elif index == 1:
            if self._params[1] > value:
                return False
        else:
            return False

        return value >= 0 and value <= 1

    def __deepcopy__(self, memo={}):
        new_object = RectangularMF(left=self._params[0], right=self._params[1])
        memo[id(self)] = new_object
        return new_object

    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self):
        return np.array([
            [0, 0],
            [self._params[0], 1],
            [self._params[1], 1],
            [1, 0],
        ])