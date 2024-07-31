import copy

import numpy as np
cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern
import cython

cdef class Dataset:
    def __init__(self, int size, int n_dim, int c_num, Pattern[:] patterns):
        """ Constructor of the class Dataset

        Args:
            size (int): Number of patterns in the dataset (used, like the other parameters, to check if the file was loaded properly)
            n_dim (int): Number of attributes (dimensions) in all the patterns of this dataset
            c_num (int): Number of class in the dataset
            patterns (Patterns[:]): Array of patterns in the dataset
        """
        if size <= 0 or n_dim <= 0 or c_num <= 0 or patterns is None:
            raise ValueError("Incorrect input dataset information")
        if size != patterns.shape[0]:
            raise ValueError("Size is not equal to the length of the patterns array")
        cdef Pattern p = patterns[0]
        if n_dim != p.get_num_dim():
            raise ValueError("The number of dimensions is not equal to the number of dimensions of the patterns")

        self.__size = size
        self.__num_dim = n_dim
        self.__num_classes = c_num
        self.__patterns = patterns

    cpdef Pattern get_pattern(self, int index):
        """Get the pattern at the given index in the dataset

        Args:
            index (int): Index of the pattern to be fetched 

        Returns:
            (Pattern) 
        """
        return self.__patterns[index]

    cpdef Pattern[:] get_patterns(self):
        """Get all the patterns in the dataset

        Returns:
            (Pattern[:]) The patterns in the dataset
        """
        return self.__patterns

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        if len(self.__patterns) == 0:
            return "null"
        txt = f"{self.__size}, {self.__num_dim}, {self.__num_classes}\n"
        for pattern in self.__patterns:
            txt += f"{pattern}\n"
        return txt

    cpdef int get_num_dim(self):
        """Get the number of dimensions (attributes) of the patterns in this dataset

        Returns:
            (int) Number of dimensions
        """
        return self.__num_dim

    cpdef int get_num_classes(self):
        """Get the number of classes in this dataset

        Returns:
            (int) Number of classes
        """
        return self.__num_classes

    cpdef int get_size(self):
        """Get the number of patterns in this dataset

        Returns:
            (int) Number of patterns
        """
        return self.__size

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            Deep copy of this object
        """
        cdef Pattern[:] patterns_copy = np.empty(self.__size, dtype=object)
        for i in range(self.__size):
            patterns_copy[i] = copy.deepcopy(self.__patterns[i])

        cdef Dataset new_object = Dataset(self.__size, self.__num_dim, self.__num_classes, patterns_copy)
        memo[id(self)] = new_object
        return new_object

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        return (self.__size == other.get_size() and
            self.__num_dim == other.get_num_dim() and
            self.__num_classes == other.get_num_classes() and
            np.array_equal(self.__patterns, other.get_patterns()))
