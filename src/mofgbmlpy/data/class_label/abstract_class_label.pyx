cimport numpy as cnp
import cython

cdef class AbstractClassLabel:
    def __init__(self):
        self.__is_rejected = False

    cpdef object get_class_label_value(self):

        raise Exception("AbstractClassLabel is abstract")

    cpdef void set_class_label_value(self, object class_label):
        # with cython.gil:
        raise Exception("AbstractClassLabel is abstract")

    cpdef cnp.npy_bool is_rejected(self):
        return self.__is_rejected

    cpdef void set_rejected(self):
        self.__is_rejected = True

    def to_xml(self):
        pass
