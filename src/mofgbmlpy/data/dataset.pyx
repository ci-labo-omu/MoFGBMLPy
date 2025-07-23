# distutils: language = c++
import copy

import numpy as np
cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern
import cython

cdef class Dataset:
    """Dataset object containing patterns

    Attributes:
        __size (int): Number of patterns in the dataset (used, like the other parameters, to check if the file was loaded properly)
        __num_dim (int): Number of attributes (dimensions) in all the patterns of this dataset
        __num_classes (int): Number of class in the dataset
        __patterns (Patterns[]): Array of patterns in the dataset
    """
    def __cinit__(self, int size, int n_dim, int c_num, Pattern[:] patterns, do_init=True):
        """ Constructor of the class Dataset

        Args:
            size (int): Number of patterns in the dataset (used, like the other parameters, to check if the file was loaded properly)
            n_dim (int): Number of attributes (dimensions) in all the patterns of this dataset
            c_num (int): Number of class in the dataset
            patterns (Patterns[]): Array of patterns in the dataset
            do_init (bool): If True, the object is initialized, otherwise it is not
        """
        if not do_init:
            self.ptr = NULL
            return

        cdef vector[PatternCpp*] cpp_patterns_vector
        cdef PatternCpp* pattern_ptr = NULL

        if patterns is not None and patterns.shape[0] != 0:
            cpp_patterns_vector.reserve(patterns.shape[0])
            for i in range(patterns.shape[0]):
                pattern_ptr = patterns[i].ptr
                cpp_patterns_vector.push_back(pattern_ptr.clone())

        self.ptr = new DatasetCpp(size, n_dim, c_num, cpp_patterns_vector)

    def __dealloc__(self):
        """Destructor"""
        if self.ptr != NULL:
            del self.ptr

    cpdef Pattern get_pattern(self, int index):
        """Get the pattern at the given index in the dataset

        Args:
            index (int): Index of the pattern to be fetched 

        Returns:
            Pattern: Pattern at the given index
        """
        return Pattern.wrap(self.ptr.get_pattern(index))

    cpdef Pattern[:] get_patterns(self):
        """Get all the patterns in the dataset

        Returns:
            Pattern[]: The patterns in the dataset
        """
        cdef vector[PatternCpp*] cpp_patterns = self.ptr.get_patterns()
        patterns = np.empty(cpp_patterns.size(), dtype=object)
        for i in range(len(patterns)):
            patterns[i] = Pattern.wrap(cpp_patterns[i])

        return patterns

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            str: String representation
        """
        return self.ptr.to_string().decode('utf-8')

    cpdef int get_num_dim(self):
        """Get the number of dimensions (attributes) of the patterns in this dataset

        Returns:
            int: Number of dimensions
        """
        return self.ptr.get_num_dim()

    cpdef int get_num_classes(self):
        """Get the number of classes in this dataset

        Returns:
            int: Number of classes
        """
        return self.ptr.get_num_classes()

    cpdef int get_size(self):
        """Get the number of patterns in this dataset

        Returns:
            int: Number of patterns
        """
        return self.ptr.get_size()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = Dataset.wrap(self.ptr)
        memo[id(self)] = new_object
        return new_object

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            bool: True if they are equal and False otherwise
        """
        if not isinstance(other, Dataset):
            return False

        cdef Dataset other_c = <Dataset> other
        return self.ptr[0] == other_c.ptr[0]

    def __len__(self):
        """Get the number of patterns in this dataset

        Returns:
            int: Number of patterns
        """
        return self.get_size()

    @staticmethod
    cdef Dataset wrap(DatasetCpp * wrapped_ptr):
        if wrapped_ptr == NULL:
            raise ValueError("pointer is NULL")
        cdef Dataset new_object = Dataset(0, 0, 0, None, do_init=False)
        new_object.ptr = wrapped_ptr.clone()
        return new_object
