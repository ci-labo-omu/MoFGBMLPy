import xml.etree.cElementTree as xml_tree
import copy
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
cimport numpy as cnp

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.exception.abstract_method_exception import AbstractMethodException
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic

cdef class AbstractConsequent:
    """Abstract class for consequent part of a fuzzy rule """

    cpdef AbstractClassLabel get_class_label(self):
        """Get the class label

        Returns:
            AbstractClassLabel: Class label
        """
        raise AbstractMethodException()

    cpdef void set_class_label_value(self, object class_label_value):
        """Set the class label value of the class label object

        Args:
            class_label_value (object): New class label value
        """
        self.get_class_label().set_class_label_value(class_label_value)

    cpdef object get_class_label_value(self):
        """Get the class label value of the class label object

        Returns:
            object: Class label value
        """
        return self.get_class_label().get_class_label_value()

    cpdef bint is_rejected(self):
        """Check if the consequent is rejected
        
        Returns:
            bool: Returns true if it is rejected and false otherwise
        """
        return self.get_class_label().is_rejected()

    cpdef void set_rejected(self):
        """Set the consequent as being rejected """
        self.get_class_label().set_rejected()

    cpdef AbstractRuleWeight get_rule_weight(self):
        """Get the rule weight object

        Returns:
            AbstractRuleWeight: Rule weight object
        """
        raise AbstractMethodException()

    cpdef void set_rule_weight(self, AbstractRuleWeight rule_weight):
        """Set the rule weight object

        Args:
            rule_weight (AbstractRuleWeight): New rule weight object
        """
        raise AbstractMethodException()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        raise AbstractMethodException()

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"class:[{self.get_class_label()}]: weight:[{self.get_rule_weight()}]"

    def __eq__(self, other):
        """Check if another object is equal to this one

        Args:
            other (object): Object compared to this one

        Returns:
            (bool) True if they are equal and False otherwise
        """
        raise AbstractMethodException()

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("consequent")
        root.append(self.get_class_label().to_xml())
        root.append(self.get_rule_weight().to_xml())

        return root

    cpdef str get_linguistic_representation(self):
        """Get the linguistic representation of the consequent
        
        Returns:
            str: Linguistic representation
        """
        return "Class is " + str(self.get_class_label()) + " with RW: " + str(self.get_rule_weight())
