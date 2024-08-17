import xml.etree.cElementTree as xml_tree
import numpy as np

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp


cdef class RuleWeightMulti(AbstractRuleWeight):
    """Rule weight for multilabel classification

    Attributes:
        __rule_weight (double[]): Value of the rule weight
    """
    def __init__(self, double[:] rule_weight):
        """Constructor

        Args:
            rule_weight (double[]): Value of the rule weight
        """
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
        """Get the rule weight for the class at the given index
        
        Args:
            index (int): Index of the class whose rule weight value is fetched 

        Returns:
            double: Rule weight value
        """
        return self.get_value()[index]

    cpdef int get_length(self):
        """Get the length of this rule weight object (number of classes)
        
        Returns:
            int: Length
        """
        return self.get_value().shape[0]

    cpdef object get_value(self):
        """Get the rule weight value
        
        Returns:
            double[]: Rule weight value
        """
        return self.__rule_weight

    cpdef void set_value(self, object rule_weight):
        """Set the value of the rule weight

        Args:
            rule_weight (double[]): New rule weight value
        """
        self.__rule_weight = rule_weight

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, RuleWeightMulti):
            return False

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
        """Get the mean of all the rule weight values
        
        Returns:
            double: Mean value
        """
        cdef int i
        cdef double sum = 0
        cdef int arr_length = self.__rule_weight.shape[0]

        for i in range(arr_length):
            sum += self.__rule_weight[i]
        return sum/arr_length