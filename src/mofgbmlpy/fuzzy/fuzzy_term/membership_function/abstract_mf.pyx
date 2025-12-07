import xml.etree.cElementTree as xml_tree

import numpy as np
from mofgbmlpy.exception.abstract_method_exception import AbstractMethodException

cdef class AbstractMF:
    """Abstract membership function
        _params (float[]): List of parameters values
        _are_params_points_flag (bool): If true then we can move them in an interactive plot. e.g. for gaussian it's set to false
    """
    def __init__(self, float[:] params, bint are_params_points_flag=True):
        self._params = params
        if self._params is None:
            self._params = np.empty(0, dtype=np.float32)
        self._are_params_points_flag = are_params_points_flag

    cdef float get_value(self, float x):
        """Get membership value (accessible only from Cython code)
        
        Args:
            x (float): Value whose membership value is calculated

        Returns:
            float: Membership value
        """
        raise AbstractMethodException()

    def get_value_py(self, float x):
        """Get membership value

        Args:
            x (float): Value whose membership value is calculated

        Returns:
            float: Membership value
        """

        return self.get_value(x)

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "Abstract membership function"

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("parameterSet")

        if len(self._params) != 0:
            for i in range(len(self._params)):
                param = xml_tree.SubElement(root, "parameter")
                param.text = str(self._params[i])
                param.set("id", str(i))

        return root

    cpdef cnp.ndarray[float, ndim=1] get_params(self):
        """Get the parameters of this function (as a numpy array)
        
        Returns:
            float[]: Parameters
        """
        return np.array(self._params, dtype=np.float32)

    cpdef cnp.ndarray[float, ndim=1] get_param_range(self, int index, float x_min=0, float x_max=1):
        """Get the range of acceptable values a given parameter as a numpy array of two values
        
        Args:
            index (int): Index of the parameter whose range is got
            x_min (float): Min value of the domain for the x axis (e.g. if index is 0 for the triangular set then we get [xmin, center]
            x_max (float): Max value of the domain for the x axis (e.g. if index is 2 for the triangular set then we get [center, max]

        Returns:
            float[]: Range of possible values
        """
        raise AbstractMethodException()

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
        cdef float[:] val_range = self.get_param_range(index, x_min, x_max)
        return val_range[0] <= value and value <= val_range[1]
    
    cpdef bint are_params_points(self):
        """Check if this function parameters represent points (It is true for triangular membership functions but not gaussian ones)
        
        Returns:
            bool: True if the parameters represent points and false otherwise
        """
        return self._are_params_points_flag

    cpdef void set_param_value(self, int index, float value, float x_min=0, float x_max=1):
        """Set the value of the parameter at a give index and check beforehand if it is valid
        
        Args:
            index (int): Index of the parameter
            value (float): Value that is changed
            x_min (float): Min value of the domain for the x axis
            x_max (float): Max value of the domain for the x axis

        Raises:
            Exception: The index is out of bounds or the value is invalid for the corresponding parameter
        """
        if self.is_param_value_valid(index, value, x_min, x_max):
            self._params[index] = value
        else:
            raise ValueError("Invalid index or value")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        raise AbstractMethodException()

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, self.__class__):
            return False
        return np.array_equal(self._params, other.get_params())

    cpdef cnp.ndarray[float, ndim=2] get_plot_points(self, float x_min=0, float x_max=1):
        """Get the plot points coordinates
        
        Args:
            x_min (float): Min value of the domain for the x axis
            x_max (float): Max value of the domain for the x axis
        Returns:
            Points coordinates that define this function shape
        """
        raise AbstractMethodException()

    cpdef float get_support(self, float x_min=0, float x_max=0):
        """Get the support value associated to this function: area covered by this function in the space "domain x [0, 1]"
        
        Args:
            x_min (float): Min value of the domain for the x axis
            x_max (float): Max value of the domain for the x axis
            
        Returns:
            Support value
        """
        raise AbstractMethodException()