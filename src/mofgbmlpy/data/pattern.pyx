import copy
import numpy as np
cimport numpy as cnp
import cython
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic

cdef class Pattern:
    """Pattern (row) of a dataset. Contains a vector of attributes and a class label

    Attributes:
        __id (int): ID of the pattern
        __attributes_vector (double[]): Array of the attributes. The size of this array is the number of dimensions
        __target_class (AbstractClassLabel): Class label associated to this pattern
    """

    def __init__(self, int pattern_id, double[:] attributes_vector, AbstractClassLabel target_class):
        """Constructor

        Args:
            pattern_id (int): ID of the pattern
            attributes_vector (double[]): Array of the attributes. The size of this array is the number of dimensions
            target_class (AbstractClassLabel): Class label associated to this pattern
        """
        if pattern_id < 0:
            raise ValueError('id must be positive')
        elif attributes_vector is None:
            raise TypeError('attribute_vector must not be None')
        elif target_class is None:
            raise TypeError('target_class must not be None')

        self.__id = pattern_id
        self.__attributes_vector = attributes_vector
        self.__target_class = target_class

    cpdef int get_id(self):
        """Get the ID
        
        Returns:
            int: ID of the pattern
        """
        return self.__id

    cpdef double[:] get_attributes_vector(self):
        """Get the attributes vector
        
        Returns:
            double[]: Array of attributes values
        """
        return self.__attributes_vector

    cpdef double get_attribute_value(self, int index):
        """Get the attribute value at the given index
        
        Args:
            index (int): Index of the attribute whose value is returned 

        Returns:
            double: Attribute value
        """
        if index < 0 or index >= self.__attributes_vector.shape[0]:
            Exception("Index is out of bounds")
        return self.__attributes_vector[index]

    cpdef object get_target_class(self):
        """Get the target class label of this pattern
        
        Returns:
            object: Target class label. Either a int or an array of int (multi label)
        """
        return self.__target_class

    cpdef int get_num_dim(self):
        """Get the number of dimensions of the attribute vector
        
        Returns:
            int: Number of dimensions
        """
        return len(self.__attributes_vector)

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            str: String representation
        """
        if self.get_attributes_vector() is None or self.get_target_class() is None:
            return "null"

        return f"[id:{self.get_id()}, input:{{{np.asarray(self.get_attributes_vector())}}}, Class:{self.get_target_class()}]"

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef double[:] vector_copy = np.copy(self.__attributes_vector)
        cdef Pattern new_object = Pattern(self.__id, vector_copy, copy.deepcopy(self.__target_class))
        memo[id(self)] = new_object
        return new_object

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            bool: True if they are equal and False otherwise
        """
        if not isinstance(other, Pattern):
            return False

        return (self.__id == other.get_id() and
                self.__target_class == other.get_target_class() and
                np.array_equal(self.__attributes_vector, other.get_attributes_vector()))
