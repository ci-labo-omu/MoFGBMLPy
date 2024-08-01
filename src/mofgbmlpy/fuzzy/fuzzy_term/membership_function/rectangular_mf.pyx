import xml.etree.cElementTree as xml_tree

import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF

cdef class RectangularMF(AbstractMF):
    def __init__(self, left=0, right=1):
        if left > right:
            raise Exception(f"Error in triangular membership function: left={left:.2f} should be < right={right:.2f}")
        super().__init__(np.array([left, right], dtype=np.float64))

    cdef double get_value(self, double x):
        return 1 if x >= self._params[0] and x <= self._params[1] else 0

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "<Rectangular MF>"

    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index, double x_min=0, double x_max=1):
        if x_min > self._params[0] or x_max < self._params[1]:
            raise Exception(f"Invalid x_min or x_max. They must be in the range [{self._params[0]}, {self._params[1]}]")

        if index == 0:
            return np.array([x_min, self._params[1]])
        elif index == 1:
            return np.array([self._params[0], x_max])
        else:
            raise Exception("Invalid index for rectangular MF")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            (object) Deep copy of this object
        """
        new_object = RectangularMF(left=self._params[0], right=self._params[1])
        memo[id(self)] = new_object
        return new_object

    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self, double x_min=0, double x_max=1):
        return np.array([
            [0, 0],
            [self._params[0], 1],
            [self._params[1], 1],
            [1, 0],
        ])