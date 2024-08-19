import xml.etree.cElementTree as xml_tree

import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF

cdef class RectangularMF(AbstractMF):
    """Rectangular membership function"""
    def __init__(self, left=0, right=1):
        """Constructor

        Args:
            left (double): X coordinate of the leftmost side of the rectangle: membership is equals to 0 before this point and 1 after it
            right (double): X coordinate of the leftmost side of the rectangle: membership is equals to 0 after this point and 1 before it
        """
        if left > right:
            raise ValueError(f"Error in triangular membership function: left={left:.2f} should be < right={right:.2f}")
        super().__init__(np.array([left, right], dtype=np.float64))

    cdef double get_value(self, double x):
        """Get membership value (accessible only from Cython code)

        Args:
            x (double): Value whose membership value is calculated

        Returns:
            double: Membership value
        """
        return 1 if x >= self._params[0] and x <= self._params[1] else 0

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "<Rectangular MF>"

    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index, double x_min=0, double x_max=1):
        """Get the range of acceptable values a given parameter as a numpy array of two values

        Args:
            index (int): Index of the parameter whose range is got
            x_min (double): Min value of the domain for the x axis (e.g. if index is 0 for the triangular set then we get [xmin, center]
            x_max (double): Max value of the domain for the x axis (e.g. if index is 2 for the triangular set then we get [center, max]

        Returns:
            double[]: Range of possible values
        """
        if x_min > self._params[0] or x_max < self._params[1]:
            raise ValueError(f"Invalid x_min or x_max. They must be in the range [{self._params[0]}, {self._params[1]}]")

        if index == 0:
            return np.array([x_min, self._params[1]])
        elif index == 1:
            return np.array([self._params[0], x_max])
        else:
            raise IndexError("Invalid index for rectangular MF")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = RectangularMF(left=self._params[0], right=self._params[1])
        memo[id(self)] = new_object
        return new_object

    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self, double x_min=0, double x_max=1):
        """Get the plot points coordinates
        
        Args:
           x_min (double): Min value of the domain for the x axis
           x_max (double): Max value of the domain for the x axis
        Returns:
           Points coordinates that define this function shape
        """
        return np.array([
            [x_min, 0],
            [self._params[0], 1],
            [self._params[1], 1],
            [x_max, 0],
        ], np.float64)

    cpdef double get_support(self, double x_min=0, double x_max=0):
        """Get the support value associated to this function: area covered by this function in the space "domain x [0, 1]"

        Args:
            x_min (double): Min value of the domain for the x axis
            x_max (double): Max value of the domain for the x axis

        Returns:
            Support value
        """
        return self._params[1] - self._params[0]
