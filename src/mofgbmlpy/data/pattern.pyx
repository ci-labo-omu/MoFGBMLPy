import copy
import numpy as np
cimport numpy as cnp
import cython
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic

cdef class Pattern:
    def __init__(self, int pattern_id, double[:] attributes_vector, AbstractClassLabel target_class):
        if pattern_id < 0:
            raise ValueError('id must be positive')
        elif attributes_vector is None:
            raise ValueError('attribute_vector must not be None')
        elif target_class is None:
            raise ValueError('target_class must not be None')

        self.__id = pattern_id
        self.__attributes_vector = attributes_vector
        self.__target_class = target_class

    cpdef int get_id(self):
        return self.__id

    cpdef double[:] get_attributes_vector(self):
        return self.__attributes_vector

    cpdef double get_attribute_value(self, int index):
        return self.__attributes_vector[index]

    cpdef object get_target_class(self):
        return self.__target_class

    cpdef int get_num_dim(self):
        return len(self.__attributes_vector)

    def __str__(self):
        if self.get_attributes_vector() is None or self.get_target_class() is None:
            return "null"

        return f"[id:{self.get_id()}, input:{{{self.get_attributes_vector()}}}, Class:{self.get_target_class()}]"

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            Deep copy of this object
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
            (bool) True if they are equal and False otherwise
        """
        return (self.__id == other.get_id() and
                self.__target_class == other.get_target_class() and
                np.array_equal(self.__attributes_vector, other.get_attributes_vector()))
