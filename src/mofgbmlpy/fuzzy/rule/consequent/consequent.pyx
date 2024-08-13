import xml.etree.cElementTree as xml_tree
import copy
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
cimport numpy as cnp


cdef class Consequent:
    def __init__(self, class_label, rule_weight):
        self._class_label = class_label
        self._rule_weight = rule_weight

    cpdef AbstractClassLabel get_class_label(self):
        return self._class_label

    cpdef void set_class_label_value(self, object class_label_value):
        self._class_label.set_class_label_value(class_label_value)

    cpdef object get_class_label_value(self):
        return self._class_label.get_class_label_value()

    cpdef bint is_rejected(self):
        return self.get_class_label().is_rejected()

    cpdef void set_rejected(self):
        self._class_label.set_rejected()

    cpdef object get_rule_weight(self):
        return self._rule_weight

    cpdef void set_rule_weight(self, object rule_weight):
        self._rule_weight = rule_weight

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_consequent = Consequent(copy.deepcopy(self._class_label), copy.deepcopy(self._rule_weight))
        memo[id(self)] = new_consequent
        return new_consequent

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"class:[{self._class_label}]: weight:[{self._rule_weight}]"

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, Consequent):
            return False

        return self._class_label == other.get_class_label() and self._rule_weight == other.get_rule_weight()

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("consequent")
        root.append(self._class_label.to_xml())
        root.append(self._rule_weight.to_xml())

        return root

    cpdef str get_linguistic_representation(self):
        return "Class is " + str(self._class_label)
