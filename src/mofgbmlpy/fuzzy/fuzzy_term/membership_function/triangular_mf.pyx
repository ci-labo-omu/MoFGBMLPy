import xml.etree.cElementTree as xml_tree

import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
cimport numpy as cnp


cdef class TriangularMF(AbstractMF):
    def __init__(self, left=0, center=0.5, right=1):
        if left > center:
            # with cython.gil:
            raise Exception(f"Error in triangular membership function: left={left:.2f} should be <= center={center:.2f}")
        elif center > right:
            # with cython.gil:
            raise Exception(f"Error in triangular membership function: center={center:.2f} should be <= right={right:.2f}")

        super().__init__(np.array([left,center,right]))

    cdef double get_value(self, double x):
        if x == self._params[0]:
            return 1
        if x <= self._params[0] or x >= self._params[2]:
            return 0
        elif x < self._params[1]:
            return (x - self._params[0]) / (self._params[1] - self._params[0])
        else:
            return (self._params[2] - x) / (self._params[2] - self._params[1])

    def __str__(self):
        return "<Triangular MF (%f, %f, %f)>" % (self._params[0], self._params[1], self._params[2])


    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index):
        if index == 0:
            return np.array([0, self._params[1]])
        elif index == 1:
            return np.array([self._params[1], self._params[2]])
        elif index == 2:
            return np.array([self._params[2], 1])
        else:
            raise Exception("Invalid index for rectangular MF")

    cpdef bint is_param_value_valid(self, int index, double value):
        if index < 0 or index >= 3:
            return False

        if index == 0:
            if value > self._params[1]:
                return False
        elif index == 1:
            if value < self._params[0] or self._params[2] < value:
                return False
        else:
            if self._params[1] > value:
                return False

        return value >= 0 and value <= 1
    
    def __deepcopy__(self, memo={}):
        new_object = TriangularMF(left=self._params[0], center=self._params[1], right=self._params[2])
        memo[id(self)] = new_object
        return new_object

    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self):
        points = []
        if self._params[0] != 0:
            points.append([0,0])
        points.append([self._params[0],0])
        points.append([self._params[1],1])
        points.append([self._params[2],0])
        if self._params[2] != 1:
            points.append([1,0])
        return np.array(points)