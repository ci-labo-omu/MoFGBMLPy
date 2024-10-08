import xml.etree.cElementTree as xml_tree
import copy
import numpy as np

from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
cimport numpy as cnp
import cython


cdef class ClassLabelMulti(AbstractClassLabel):
    """Class label class for multi labels classification (array of integers)

    Attributes:
        __class_label (int[]): Values associated to class labels
    """

    def __init__(self, int[:] class_label):
        """Constructor

        Args:
            class_label (int[]): Class label values
        """
        if class_label is None:
            self.__class_label = np.empty(0, int)
        else:
            self.__class_label = class_label
        super().__init__()

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            bool: True if they are equal and False otherwise
        """
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
        """Returns the length of the array of class label values
        
        Returns:
            int: Length of the class label values array
        """
        return self.__class_label.shape[0]

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef int[:] value_copy = None
        if self.__class_label is not None:
            value_copy = np.copy(self.__class_label)

        cdef ClassLabelMulti new_object = ClassLabelMulti(value_copy)
        memo[id(self)] = new_object
        return new_object

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            str: String representation
        """
        cdef int[:] label_value = self.__class_label
        txt = f"{label_value[0]:2d}"

        if self.get_length() > 1:
            for i in range(1, self.get_length()):
                txt = f"{txt}, {label_value[i]:2d}"

        return txt

    cpdef object get_class_label_value(self):
        """Get the class label values

        Returns:
            int[]: Class label values
        """
        return self.__class_label

    cpdef int get_class_label_value_at(self, int index):
        """Get the class label value at the given index

        Returns:
            int: Class label value
        """
        if index < 0 or index > self.__class_label.shape[0]:
            raise IndexError("index is out of bounds for the class label object")

        return self.__class_label[index]

    cpdef void set_class_label_value(self, object class_label):
        """Set the class label values

            Args:
                class_label (int[]): New class label values 
            """
        if class_label is None:
            raise TypeError("class_label can't be None")
        cdef int[:] value = class_label
        self.__class_label = class_label

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            xml.etree.ElementTree: XML element representing this object
        """
        root = xml_tree.Element("classLabel")
        root.text = str(self)

        return root
