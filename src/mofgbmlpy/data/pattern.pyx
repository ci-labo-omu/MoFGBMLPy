from copy import deepcopy
import numpy as np
cimport numpy as cnp


cdef class Pattern:
    # cdef int __id
    # cdef object __attribute_vector
    # cdef object __target_class

    def __init__(self, pattern_id, attribute_vector, target_class):
        if pattern_id < 0:
            raise ValueError('id must be positive')
        elif attribute_vector is None:
            raise ValueError('attribute_vector must not be None')
        elif target_class is None:
            raise ValueError('target_class must not be None')

        self.__id = pattern_id
        self.__attribute_vector = attribute_vector
        self.__target_class = target_class

    cpdef int get_id(self):
        return self.__id

    cpdef cnp.ndarray[object, ndim=1] get_attributes_vector(self):
        return self.__attribute_vector

    cpdef double get_attribute_value(self, int index):
        return self.__attribute_vector[index]

    cpdef object get_target_class(self):
        return self.__target_class

    def __str__(self):
        if self.get_attributes_vector() is None or self.get_target_class() is None:
            return "null"

        return f"[id:{self.get_id()}, input:{{{self.get_attributes_vector()}}}, Class:{self.get_target_class()}]"

    # def __reduce__(self):
    #     cdef cnp.ndarray[double, ndim=1] vector_copy = np.empty(self.__attribute_vector.shape[0])
    #     for i in range(self.__attribute_vector.shape[0]):
    #         vector_copy[i] = self.__attribute_vector[i]
    #     return (self.__class__, (self.__id, vector_copy, self.__target_class))
