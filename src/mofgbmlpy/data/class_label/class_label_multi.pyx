# distutils: language = c++

import xml.etree.cElementTree as xml_tree
import copy
import numpy as np

from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel, AbstractClassLabelCpp
from mofgbmlpy.data.class_label.class_label_multi cimport ClassLabelMultiCpp
cimport numpy as cnp
import cython
from libcpp.vector cimport vector


cdef class ClassLabelMulti(AbstractClassLabel):
    """Class label class for multi labels classification (array of integers)

    Attributes:
        __class_label (int[]): Values associated to class labels
    """

    def __cinit__(self, int[:] class_label=None, do_init=True):
        """Constructor

        Args:
            class_label (int[]): Class label values
            do_init (bool): If True, the object is initialized, otherwise it is not
        """
        if not do_init:
            self.ptr = NULL
            return

        cdef vector[int] cpp_vector
        if class_label is None:
            self.ptr = new ClassLabelMultiCpp(cpp_vector)
        else:
            cpp_vector.reserve(class_label.shape[0])
            for i in range(class_label.shape[0]):
                cpp_vector.push_back(class_label[i])
            self.ptr = new ClassLabelMultiCpp(cpp_vector)

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            bool: True if they are equal and False otherwise
        """
        if not isinstance(other, ClassLabelMulti):
            return False

        cdef ClassLabelMulti other_c = <ClassLabelMulti> other
        return other_c.get_multi_ptr()[0] == self.get_multi_ptr()[0]

    cpdef int get_length(self):
        """Returns the length of the array of class label values
        
        Returns:
            int: Length of the class label values array
        """
        return self.get_multi_ptr().get_length()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = ClassLabelMulti.wrap(self.get_multi_ptr())
        memo[id(self)] = new_object
        return new_object

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            str: String representation
        """
        return self.get_multi_ptr().to_string().decode('utf-8')

    cpdef object get_class_label_value(self):
        """Get the class label values

        Returns:
            int[]: Class label values
        """
        return np.array(self.get_multi_ptr().get_class_label_value(), dtype=np.int32)

    cpdef int get_class_label_value_at(self, int index):
        """Get the class label value at the given index

        Returns:
            int: Class label value
        """
        return self.get_multi_ptr().get_class_label_value_at(index)

    cpdef void set_class_label_value(self, object class_label):
        """Set the class label values

            Args:
                class_label (int[]): New class label values 
            """
        if isinstance(class_label, np.ndarray) and class_label.dtype != np.int32:
            raise ValueError("Class label values must be of type int32")
        self.get_multi_ptr().set_class_label_value(class_label)

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            xml.etree.ElementTree: XML element representing this object
        """
        root = xml_tree.Element("classLabel")
        root.text = str(self)

        return root

    cdef ClassLabelMultiCpp * get_multi_ptr(self):
        return <ClassLabelMultiCpp*> self.ptr

    @staticmethod
    cdef ClassLabelMulti wrap(ClassLabelMultiCpp * wrapped_ptr):
        if wrapped_ptr == NULL:
            raise ValueError("pointer is NULL")

        cdef ClassLabelMulti new_object = ClassLabelMulti(do_init=False)
        new_object.ptr = wrapped_ptr.clone()
        return new_object

    cpdef void set_class_label_value_at(self, int index, int value):
        self.get_multi_ptr().set_class_label_value_at(index, value)