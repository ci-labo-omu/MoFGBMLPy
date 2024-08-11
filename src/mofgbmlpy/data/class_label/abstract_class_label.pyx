cimport numpy as cnp
import cython

cdef class AbstractClassLabel:
    """Abstract class for class labels

    Attributes
        __is_rejected (bool): If True then the class label is rejected (it can't be used for classification)
    """

    def __init__(self):
        """Constructor of this class. Initialize is_rejected to False
        """
        self.__is_rejected = False

    cpdef object get_class_label_value(self):
        """Get the class label value (array of int if it's a multilabel and int if it's not). Must be overridden.
        
        Returns:
            object: Class label value
        """
        raise Exception("AbstractClassLabel is abstract")

    cpdef void set_class_label_value(self, object class_label):
        """Set the class label value. Must be overridden.
        
        Args:
            class_label (object): New class label value, either a int or an array of int depending on the label type (multi or nsgaii) 
        """
        raise Exception("AbstractClassLabel is abstract")

    cpdef bint is_rejected(self):
        """Check if the class label is rejected
        
        Returns:
            bool: True if it's rejected and False otherwise
        """
        return self.__is_rejected

    cpdef void set_rejected(self):
        """Set this class label to "rejected"
        
        """
        self.__is_rejected = True

    def to_xml(self):
        """Get the XML representation of this object. Must be overridden.

        Returns:
            :xml.etree.ElementTree: XML element representing this object
        """
        raise Exception("AbstractClassLabel is abstract")
