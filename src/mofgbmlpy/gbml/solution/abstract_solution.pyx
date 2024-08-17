from abc import ABC, abstractmethod
import numpy as np
cimport numpy as cnp


cdef class AbstractSolution:
    """Abstract solution

    Attributes:
        _attributes (dict): Dictionary used to save data into this solution. It's not used during training or eval
        _objectives (float[]): Objective values for this solution
    """
    def __init__(self, num_objectives, num_constraints=0):
        """Constructor

        Args:
            num_objectives (int): Number of objectives
            num_constraints (int): Number of constraints (not used in the current version
        """
        self._attributes = {}
        self._objectives = np.zeros(num_objectives, dtype=np.float64)
        # self.__constraints = np.empty(num_constraints, dtype=np.float64)

    cpdef double[:] get_objectives(self):
        """Get the array of objectives
        
        Returns:
            double[]: Objectives
        """
        return self._objectives

    # cpdef double[:] get_constraints(self):
    #     return self.__constraints

    cpdef void set_attribute(self, str key, object value):
        """Set the attribute key to the give nvalue
        
        Args:
            key (str): Key added or changed
            value (object): New value 

        """
        self._attributes[key] = value

    cpdef object get_attribute(self, str key):
        """Get the attribute value at the given key
        
        Args:
            key (str): Key whose corresponding value is fetched 

        Returns:
            object: Attribute value fetched
        """
        return self._attributes[key]

    cpdef bint has_attribute(self, str key):
        """Check if the attribute exists
        
        Args:
            key (str): Key searched 

        Returns:
            bool: True if it exists and false otherwise
        """
        return key in self._attributes

    cpdef void set_objective(self, int index, double value):
        """Set the objective at the given index
        
        Args:
            index (int): Index of the objective whose value is changed
            value (double): New value 
        """
        self._objectives[index] = value

    cpdef double get_objective(self, int index):
        """Get the objective at the given index
        
        Args:
            index (int): Index where the objective is fetched

        Returns:
            double: Objective value fetched
        """
        return self._objectives[index]

    cpdef int get_num_vars(self):
        """Get the number of variables
        
        Returns:
            int: Number of variables
        """
        raise Exception("This class is abstract")

    cdef void clear_vars(self):
        """Clear the variables"""
        raise Exception("This class is abstract")

    # cpdef double get_constraint(self, int index):
    #     return self.__constraints[index]
    #
    # cpdef void set_constraint(self, int index, double value):
    #     self.__constraints[index] = value

    cpdef int get_num_objectives(self):
        """Get the number of objectives
        
        Returns:
            int: Number of objectives
        """
        return self._objectives.shape[0]

    cpdef int get_num_constraints(self):
        """Get the number of constraints
        
        Returns:
            int: Number of constraints
        """
        # return self.__constraints.shape[0]
        # TODO
        return 0

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        raise Exception("This class is abstract")

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        raise Exception("This class is abstract")

    def __hash__(self):
        """Hash function

        Returns:
            object: Hash value
        """
        raise Exception("This class is abstract")

    cpdef object get_attributes(self):
        """get the attributes dictionary
        
        Returns:
            dict: Attributes
        """
        return self._attributes

    cpdef void clear_attributes(self):
        """Clear the attributes"""
        self._attributes = {}
