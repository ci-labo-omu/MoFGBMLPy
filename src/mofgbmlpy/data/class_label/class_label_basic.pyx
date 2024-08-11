import xml.etree.cElementTree as xml_tree
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
import cython


cdef class ClassLabelBasic(AbstractClassLabel):
    """Class label class for single label classification (one integer associated to a class)

    Attributes:
        __class_label (int): Class label
    """

    def __init__(self, int class_label):
        """Constructor

        Args:
            class_label (int): Class label value
        """
        super().__init__()
        self.__class_label = class_label

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            bool: True if they are equal and False otherwise
        """
        if not isinstance(other, ClassLabelBasic):
            return False
        return other.get_class_label_value() == self.__class_label

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = ClassLabelBasic(self.__class_label)
        memo[id(self)] = new_object
        return new_object

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            str: String representation
        """
        if self.__class_label is None:
            raise Exception("class label value is None")
        return f"{self.__class_label:2d}"

    cpdef object get_class_label_value(self):
        """Get the class label value

            Returns:
                int: Class label value
            """
        return self.__class_label

    cpdef void set_class_label_value(self, object class_label):
        """Set the class label value

            Args:
                class_label (int): New class label value 
            """
        self.__class_label = class_label


    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            xml.etree.ElementTree: XML element representing this object
        """
        root = xml_tree.Element("classLabel")
        root.text = str(self)

        return root
