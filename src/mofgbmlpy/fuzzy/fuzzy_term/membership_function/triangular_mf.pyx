import xml.etree.cElementTree as xml_tree

import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
cimport numpy as cnp


cdef class TriangularMF(AbstractMF):
    """Triangular membership function"""
    def __cinit__(self, left=0, center=0.5, right=1, do_init=True):
        """Constructor

        Args:
            left (float): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 before it
            center (float): X coordinate of the vertex in the center of the triangle: membership is equals to 1 at this point
            right (float): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 after it
            do_init (bool): If True, the object is initialized, otherwise it is not
        """

        if not do_init:
            self.ptr = NULL
            return

        self.ptr = new TriangularMFCpp(left, center, right)

    cdef float get_value(self, float x):
        """Get membership value (accessible only from Cython code)

        Args:
            x (float): Value whose membership value is calculated

        Returns:
            float: Membership value
        """
        return self.ptr.get_value(x)


    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return self.ptr.to_string().decode('utf-8')

    cpdef cnp.ndarray[float, ndim=1] get_param_range(self, int index, float x_min=0, float x_max=1):
        """Get the range of acceptable values a given parameter as a numpy array of two values

            Args:
                index (int): Index of the parameter whose range is got
                x_min (float): Min value of the domain for the x axis (e.g. if index is 0 for the triangular set then we get [xmin, center]
                x_max (float): Max value of the domain for the x axis (e.g. if index is 2 for the triangular set then we get [center, max]

            Returns:
                float[]: Range of possible values
            """
        return np.array(self.ptr.get_param_range(index, x_min, x_max), dtype=np.float32)


    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = TriangularMF.wrap(<TriangularMFCpp*> self.ptr)
        memo[id(self)] = new_object
        return new_object

    cpdef cnp.ndarray[float, ndim=2] get_plot_points(self, float x_min=0, float x_max=1):
        """Get the plot points coordinates

           Args:
               x_min (float): Min value of the domain for the x axis
               x_max (float): Max value of the domain for the x axis
           Returns:
               Points coordinates that define this function shape
           """

        return np.array(self.ptr.get_plot_points(x_min, x_max), dtype=np.float32)

    cpdef float get_support(self, float x_min=0, float x_max=0):
        """Get the support value associated to this function: area covered by this function in the space "domain x [0, 1]"

        Args:
            x_min (float): Min value of the domain for the x axis
            x_max (float): Max value of the domain for the x axis

        Returns:
            Support value
        """
        return self.ptr.get_support(x_min, x_max)

    @staticmethod
    cdef TriangularMF wrap(TriangularMFCpp * wrapped_ptr):
        if wrapped_ptr == NULL:
            raise ValueError("pointer is NULL")

        cdef TriangularMF new_object = TriangularMF(do_init=False)
        new_object.ptr = wrapped_ptr.clone()

        return new_object
