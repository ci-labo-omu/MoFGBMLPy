import xml.etree.cElementTree as xml_tree
cimport numpy as cnp
import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF

cdef class DontCareMF(AbstractMF):
    """Don't care membership function (returns always 1) """
    def __init__(self):
        """Constructor """
        super().__init__(None)

    cdef float get_value(self, float _):
        """Get membership value (accessible only from Cython code)
        
        Args:
            _ (float): Value whose membership value is calculated

        Returns:
            float: Membership value
        """
        return 1.0

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "<Dont Care MF>"

    cpdef cnp.ndarray[float, ndim=1] get_param_range(self, int index, float x_min=0, float x_max=1):
        """Get the range of acceptable values a given parameter as a numpy array of two values

        Args:
            index (int): Index of the parameter whose range is got
            x_min (float): Min value of the domain for the x axis (e.g. if index is 0 for the triangular set then we get [xmin, center]
            x_max (float): Max value of the domain for the x axis (e.g. if index is 2 for the triangular set then we get [center, max]

        Returns:
            float[]: Range of possible values
        """
        return np.empty(0, dtype=np.float32)

    cpdef bint is_param_value_valid(self, int index, float value, float x_min=0, float x_max=1):
        """Check if the provided value for the parameter at the given index is valid

        Args:
            index (int): Index of the parameter
            value (float): Value that is checked for the parameter
            x_min (float): Min value of the domain for the x axis
            x_max (float): Max value of the domain for the x axis

        Returns:
            bool: True if it is valid and false otherwise
        """
        return False # Can't be edited

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = DontCareMF()
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
        return np.array([[x_min,1], [x_max,1]], np.float32)

    cpdef float get_support(self, float x_min=0, float x_max=0):
        """Get the support value associated to this function: area covered by this function in the space "domain x [0, 1]"

        Args:
            x_min (float): Min value of the domain for the x axis
            x_max (float): Max value of the domain for the x axis

        Returns:
            Support value
        """
        return 1.0