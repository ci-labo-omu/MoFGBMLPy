import copy

import numpy as np
cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern
import cython

cdef class Dataset:
    def __init__(self, size, n_dim, c_num, patterns):
        if size <= 0 or n_dim <= 0 or c_num <= 0:
            # with cython.gil:
            raise ValueError("Incorrect input data set information")
        self.__size = size
        self.__num_dim = n_dim
        self.__num_classes = c_num
        self.__patterns = np.array(patterns)

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