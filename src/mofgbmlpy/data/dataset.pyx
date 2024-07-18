import copy

import numpy as np
cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern
import cython

cdef class Dataset:
    def __init__(self, int size, int n_dim, int c_num, Pattern[:] patterns):
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
        return self.__patterns[index]

    cpdef Pattern[:] get_patterns(self):
        return self.__patterns

    def __str__(self):
        if len(self.__patterns) == 0:
            return "null"
        txt = f"{self.__size}, {self.__num_dim}, {self.__num_classes}\n"
        for pattern in self.__patterns:
            txt += f"{pattern}\n"
        return txt

    cpdef int get_num_dim(self):
        return self.__num_dim

    cpdef int get_num_classes(self):
        return self.__num_classes

    cpdef int get_size(self):
        return self.__size

    def __deepcopy__(self, memo={}):
        cdef Pattern[:] patterns_copy = np.empty(self.__size, dtype=object)
        for i in range(self.__size):
            patterns_copy[i] = copy.deepcopy(self.__patterns[i])

        cdef Dataset new_object = Dataset(self.__size, self.__num_dim, self.__num_classes, patterns_copy)
        memo[id(self)] = new_object
        return new_object

    def __eq__(self, other):
        return (self.__size == other.get_size() and
            self.__num_dim == other.get_num_dim() and
            self.__num_classes == other.get_num_classes() and
            np.array_equal(self.__patterns, other.get_patterns()))
