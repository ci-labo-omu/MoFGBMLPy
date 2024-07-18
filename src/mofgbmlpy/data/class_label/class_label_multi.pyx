import xml.etree.cElementTree as xml_tree
import copy
import numpy as np

from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
cimport numpy as cnp
import cython


cdef class ClassLabelMulti(AbstractClassLabel):
    def __init__(self, int[:] class_label):
        self.__class_label = class_label
        super().__init__()

    def __eq__(self, other):
        if not isinstance(other, ClassLabelMulti):
            return False

        cdef int[:] label = self.__class_label
        cdef int[:] other_label = other.get_class_label_value()

        # XOR
        if (label is None) ^ (other_label is None):
            return False

        if label is None and other_label is None:
            return True

        if self.get_length() != other.get_length():
            return False

        cdef int i

        for i in range(self.get_length()):
            if label[i] != other_label[i]:
                return False
        return True

    cpdef int get_length(self):
        if self.__class_label is None:
            raise Exception("class label value is None")
        return self.__class_label.shape[0]

    def __deepcopy__(self, memo={}):
        cdef int[:] value_copy = None
        if self.__class_label is not None:
            value_copy = np.copy(self.__class_label)

        cdef ClassLabelMulti new_object = ClassLabelMulti(value_copy)
        memo[id(self)] = new_object
        return new_object

    def __str__(self):
        if self.__class_label is None:
            raise Exception("class label value is None")
        cdef int[:] label_value = self.__class_label
        txt = f"{self.label_value[0]:2d}"

        if self.get_length() > 1:
            for i in range(1, self.get_length()):
                txt = f"{txt}, {label_value[i]:2d}"

        return txt

    cpdef object get_class_label_value(self):
        return self.__class_label

    cpdef void set_class_label_value(self, object class_label):
        cdef int[:] value = class_label
        self.__class_label = class_label

    def to_xml(self):
        root = xml_tree.Element("classLabel")
        root.text = str(self)

        return root
