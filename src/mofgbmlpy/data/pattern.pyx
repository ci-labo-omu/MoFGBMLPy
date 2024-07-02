import copy
import numpy as np
cimport numpy as cnp
import cython

cdef class Pattern:
    def __init__(self, pattern_id, attribute_vector, target_class):
        if pattern_id < 0:
            # with cython.gil:
            raise ValueError('id must be positive')
        elif attribute_vector is None:
            # with cython.gil:
            raise ValueError('attribute_vector must not be None')
        elif target_class is None:
            # with cython.gil:
            raise ValueError('target_class must not be None')

        self.__id = pattern_id
        self.__attribute_vector = attribute_vector
        self.__target_class = target_class

    cpdef int get_id(self):
        return self.__id

    cpdef double[:] get_attributes_vector(self):
        return self.__attribute_vector

    cpdef double get_attribute_value(self, int index):
        return self.__attribute_vector[index]

    cpdef object get_target_class(self):
        return self.__target_class

    def __str__(self):
        if self.get_attributes_vector() is None or self.get_target_class() is None:
            return "null"

        return f"[id:{self.get_id()}, input:{{{self.get_attributes_vector()}}}, Class:{self.get_target_class()}]"

    def __deepcopy__(self, memo={}):
        cdef double[:] vector_copy = np.copy(self.__attribute_vector)
        cdef Pattern new_object = Pattern(self.__id, vector_copy, copy.deepcopy(self.__target_class))
        memo[id(self)] = new_object
        return new_object