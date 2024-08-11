import xml.etree.cElementTree as xml_tree
import numpy as np

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp


cdef class RuleWeightMulti(AbstractRuleWeight):
    def __init__(self, double[:] rule_weight):
        self.__rule_weight = rule_weight

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef double[:] values_copy = np.empty(self.get_length())
        cdef int i

        for i in range(values_copy.shape[0]):
            values_copy[i] = self.__rule_weight[i]

        new_object = RuleWeightMulti(values_copy)

        memo[id(self)] = new_object
        return new_object

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        if self.get_value() is None:
            # with cython.gil:
            raise ValueError("Rule weight is None")

        txt = f"{self.get_value()[0]:.4f}"

        if self.get_length() > 1:
            for i in range(1, self.get_length()):
                txt = f"{txt}, {self.get_value()[i]:.4f}"

        return txt

    cpdef get_rule_weight_at(self, int index):
        return self.get_value()[index]

    cpdef int get_length(self):
        return self.get_value().shape[0]

    cpdef object get_value(self):
        return self.__rule_weight

    cpdef void set_value(self, object rule_weight):
        self.__rule_weight = rule_weight

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        return np.array_equal(self.__rule_weight, other.get_value())

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("ruleWeight")
        root.text = str(self)

        return root

    cdef double get_mean(self):
        cdef int i
        cdef double sum = 0
        cdef int arr_length = self.__rule_weight.shape[0]

        for i in range(arr_length):
            sum += self.__rule_weight[i]
        return sum/arr_length