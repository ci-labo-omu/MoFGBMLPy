import xml.etree.cElementTree as xml_tree

import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF


cdef class TriangularMF(AbstractMF):
    def __init__(self, left=0, center=0.5, right=1):
        self.__left = left
        self.__center = center
        self.__right = right

        if left > center:
            # with cython.gil:
            raise Exception(f"Error in triangular membership function: left={left:.2f} should be <= center={center:.2f}")
        elif center > right:
            # with cython.gil:
            raise Exception(f"Error in triangular membership function: center={center:.2f} should be <= right={right:.2f}")

    cdef double get_value(self, double x):
        if x == self.__center:
            return 1
        if x <= self.__left or x >= self.__right:
            return 0
        elif x < self.__center:
            return (x - self.__left) / (self.__center - self.__left)
        else:
            return (self.__right - x) / (self.__right - self.__center)

    def __str__(self):
        return "<Triangular MF (%f, %f, %f)>" % (self.__left, self.__center, self.__right)

    def to_xml(self):
        root = xml_tree.Element("membershipFunction")
        params_set = xml_tree.SubElement(root, "parameterSet")
        i = 0
        for p in [self.__left, self.__center, self.__right]:
            param = xml_tree.SubElement(params_set, "parameterSet")
            param.text = str(p)
            param.set("id", str(i))
            i += 1

        return root

    cpdef double[:,:] get_plot_points(self):
        return np.array([[self.__left,0],[self.__center,1], [self.__right,0]], dtype=np.float64)