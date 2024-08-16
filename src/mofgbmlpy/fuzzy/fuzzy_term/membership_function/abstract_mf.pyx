import xml.etree.cElementTree as xml_tree

import numpy as np

cdef class AbstractMF:
    """Abstract membership function
        _params (double[]): List of parameters values
        _are_params_points_flag (bool): If true then we can move them in an interactive plot. e.g. for gaussian it's set to false
    """
    def __init__(self, double[:] params, bint are_params_points_flag=True):
        self._params = params
        if self._params is None:
            self._params = np.empty(0)
        self._are_params_points_flag = are_params_points_flag

    cdef double get_value(self, double x):
        """Get membership value (accessible only from Cython code)
        
        Args:
            x (double): Value whose membership value is calculated

        Returns:
            double: Membership value
        """
        raise Exception("This class is abstract")

    def get_value_py(self, double x):
        """Get membership value

        Args:
            x (double): Value whose membership value is calculated

        Returns:
            double: Membership value
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
        root = xml_tree.Element("membershipFunction")

        if len(self._params) != 0:
            params_set = xml_tree.SubElement(root, "parameterSet")

            for i in range(len(self._params)):
                param = xml_tree.SubElement(params_set, "parameterSet")
                param.text = str(self._params[i])
                param.set("id", str(i))

        return root

    cpdef cnp.ndarray[double, ndim=1] get_params(self):
        """Get the parameters of this function (as a numpy array)
        
        Returns:
            double[]: Parameters
        """
        return np.array(self._params)

    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index, double x_min=0, double x_max=1):
        """Get the range of acceptable values a given parameter as a numpy array of two values
        
        Args:
            index (int): Index of the parameter whose range is got
            x_min (double): Min value of the domain for the x axis (e.g. if index is 0 for the triangular set then we get [xmin, center]
            x_max (double): Max value of the domain for the x axis (e.g. if index is 2 for the triangular set then we get [center, max]

        Returns:
            double[]: Range of possible values
        """
        raise Exception("This class is abstract")

    cpdef bint is_param_value_valid(self, int index, double value, double x_min=0, double x_max=1):
        """Check if the provided value for the parameter at the given index is valid
        
        Args:
            index (int): Index of the parameter
            value (double): Value that is checked for the parameter
            x_min (double): Min value of the domain for the x axis
            x_max (double): Max value of the domain for the x axis

        Returns:
            bool: True if it is valid and false otherwise
        """
        cdef double[:] val_range = self.get_param_range(index, x_min, x_max)
        return val_range[0] <= value and value <= val_range[1]

    cpdef bint are_params_points(self):
        """Check if this function parameters represent points (It is true for triangular membership functions but not gaussian ones)
        
        Returns:
            bool: True if the parameters represent points and false otherwise
        """
        return self._are_params_points_flag

    cpdef void set_param_value(self, int index, double value, double x_min=0, double x_max=1):
        """Set the value of the parameter at a give index and check beforehand if it is valid
        
        Args:
            index (int): Index of the parameter
            value (double): Value that is changed
            x_min (double): Min value of the domain for the x axis
            x_max (double): Max value of the domain for the x axis

        Raises:
            Exception: The index is out of bounds or the value is invalid for the corresponding parameter
        """
        if self.is_param_value_valid(index, value, x_min, x_max):
            self._params[index] = value
        else:
            raise Exception("Invalid index or value")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        raise Exception("This class is abstract")

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

    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self, double x_min=0, double x_max=1):
        """Get the plot points coordinates
        
        Args:
            x_min (double): Min value of the domain for the x axis
            x_max (double): Max value of the domain for the x axis
        Returns:
            Points coordinates that define this function shape
        """
        raise Exception("This class is abstract")

    cpdef double get_support(self, double x_min=0, double x_max=0):
        """Get the support value associated to this function: area covered by it function in the space "domain x [0, 1]"
        
        Args:
            x_min (double): Min value of the domain for the x axis
            x_max (double): Max value of the domain for the x axis
            
        Returns:
            Support value
        """
        raise Exception("This class is abstract")