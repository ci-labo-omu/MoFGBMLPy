# distutils: language = c++

import xml.etree.cElementTree as xml_tree
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel, AbstractClassLabelCpp
from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasicCpp
import cython


cdef class ClassLabelBasic(AbstractClassLabel):
    """Class label class for single label classification (one integer associated to a class)

    Attributes:
        __class_label (int): Class label
    """

    def __cinit__(self, int class_label):
        """Constructor

        Args:
            class_label (int): Class label value
        """
        self.ptr = new ClassLabelBasicCpp(class_label)

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            bool: True if they are equal and False otherwise
        """
        if not isinstance(other, ClassLabelBasic):
            return False
        cdef ClassLabelBasic other_c = <ClassLabelBasic> other
        return other_c.get_basic_ptr() == self.get_basic_ptr()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef ClassLabelBasicCpp * ptr_copy = self.get_basic_ptr().clone()
        new_object = ClassLabelBasic.wrap(ptr_copy)
        memo[id(self)] = new_object
        return new_object

    cpdef int get_class_label_value(self):
        """Get the class label value

            Returns:
                int: Class label value
            """
        return self.get_basic_ptr().get_class_label_value()

    cpdef void set_class_label_value(self, int class_label):
        """Set the class label value

            Args:
                class_label (int): New class label value 
            """
        self.get_basic_ptr().set_class_label_value(class_label)


    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            xml.etree.ElementTree: XML element representing this object
        """
        root = xml_tree.Element("classLabel")
        root.text = str(self)

        return root

    cdef ClassLabelBasicCpp * get_basic_ptr(self):
        return <ClassLabelBasicCpp*> self.ptr

    @staticmethod
    cdef ClassLabelBasic wrap(ClassLabelBasicCpp * ptr):
        cdef ClassLabelBasic new_object = ClassLabelBasic.__new__(ClassLabelBasic)
        new_object.ptr = <AbstractClassLabelCpp *> ptr
        return new_object