# distutils: language = c++

from mofgbmlpy.exception.abstract_method_exception import AbstractMethodException
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabelCpp

cdef class AbstractClassLabel:
    """Abstract class for class labels

    Attributes
        __is_rejected (bool): If True then the class label is rejected (it can't be used for classification)
    """

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        del self.ptr


    cpdef bint is_rejected(self):
        """Check if the class label is rejected
        
        Returns:
            bool: True if it's rejected and False otherwise
        """
        return self.ptr.is_rejected()

    cpdef void set_rejected(self):
        """Set this class label to "rejected"
        
        """
        self.ptr.set_rejected()

    def to_xml(self):
        """Get the XML representation of this object. Must be overridden.

        Returns:
            :xml.etree.ElementTree: XML element representing this object
        """
        raise AbstractMethodException()

    def __repr__(self):
        """Return a string representation of this object

           Returns:
               str: String representation
           """
        return self.ptr.to_string().decode('utf-8')

    cdef AbstractClassLabelCpp * get_ptr(self):
        return self.ptr
