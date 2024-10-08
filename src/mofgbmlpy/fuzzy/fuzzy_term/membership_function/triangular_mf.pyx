import xml.etree.cElementTree as xml_tree

import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
cimport numpy as cnp


cdef class TriangularMF(AbstractMF):
    """Triangular membership function"""
    def __init__(self, left=0, center=0.5, right=1):
        """Constructor

        Args:
            left (double): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 before it
            center (double): X coordinate of the vertex in the center of the triangle: membership is equals to 1 at this point
            right (double): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 after it
        """
        if left is None or center is None or right is None:
            raise TypeError("Parameters can't be None")
        if left > center:
            raise ValueError(f"Error in triangular membership function: left={left:.2f} should be <= center={center:.2f}")
        elif center > right:
            raise ValueError(f"Error in triangular membership function: center={center:.2f} should be <= right={right:.2f}")

        super().__init__(np.array([left,center,right], dtype=np.float64))

    cdef double get_value(self, double x):
        """Get membership value (accessible only from Cython code)

        Args:
            x (double): Value whose membership value is calculated

        Returns:
            double: Membership value
        """
        if x == self._params[1]:
            # For the case where left = center or center = right
            return 1
        if x <= self._params[0] or x >= self._params[2]:
            return 0
        elif x < self._params[1]:
            return (x - self._params[0]) / (self._params[1] - self._params[0])
        else:
            return (self._params[2] - x) / (self._params[2] - self._params[1])


    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "<Triangular MF (%f, %f, %f)>" % (self._params[0], self._params[1], self._params[2])

    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index, double x_min=0, double x_max=1):
        """Get the range of acceptable values a given parameter as a numpy array of two values

            Args:
                index (int): Index of the parameter whose range is got
                x_min (double): Min value of the domain for the x axis (e.g. if index is 0 for the triangular set then we get [xmin, center]
                x_max (double): Max value of the domain for the x axis (e.g. if index is 2 for the triangular set then we get [center, max]

            Returns:
                double[]: Range of possible values
            """
        if x_min > self._params[0] or x_max < self._params[2]:
            raise ValueError(f"Invalid x_min or x_max. They must be in the range [{self._params[0]}, {self._params[2]}]")

        if index == 0:
            return np.array([x_min, self._params[1]])
        elif index == 1:
            return np.array([self._params[0], self._params[2]])
        elif index == 2:
            return np.array([self._params[1], x_max])
        else:
            raise IndexError("Invalid index for rectangular MF")


    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = TriangularMF(left=self._params[0], center=self._params[1], right=self._params[2])
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
            [x_min,0],
            [self._params[0], 0],
            [self._params[1], 1],
            [self._params[2], 0],
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
        return 0.5 * (self._params[2] - self._params[0])  # right - left