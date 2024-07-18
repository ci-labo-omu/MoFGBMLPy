import xml.etree.cElementTree as xml_tree
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
import cython


cdef class ClassLabelBasic(AbstractClassLabel):
    def __init__(self, int class_label):
        super().__init__()
        self.__class_label = class_label

    def __eq__(self, other):
        if not isinstance(other, ClassLabelBasic):
            return False
        return other.get_class_label_value() == self.__class_label

    def __deepcopy__(self, memo={}):
        new_object = ClassLabelBasic(self.__class_label)
        memo[id(self)] = new_object
        return new_object

    def __str__(self):
        if self.__class_label is None:
            raise Exception("class label value is None")
        return f"{self.__class_label:2d}"

    cpdef object get_class_label_value(self):
        return self.__class_label

    cpdef void set_class_label_value(self, object class_label):
        self.__class_label = class_label


    def to_xml(self):
        root = xml_tree.Element("classLabel")
        root.text = str(self)

        return root
