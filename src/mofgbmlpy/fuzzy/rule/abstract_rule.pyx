import xml.etree.cElementTree as xml_tree
from abc import ABC, abstractmethod

import numpy as np
cimport numpy as cnp

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight

cdef class AbstractRule:
    """Abstract fuzzy rule class

    Attributes:
        _antecedent (Antecedent): Antecedent of the rule
        _consequent (Consequent): Consequent of the rule
    """
    def __init__(self, antecedent, consequent):
        """Constructor

        Args:
            antecedent (Antecedent): Antecedent of the rule
            consequent (Consequent): Consequent of the rule
        """
        self._antecedent = antecedent
        self._consequent = consequent

    cpdef Antecedent get_antecedent(self):
        """Get the antecedent
        
        Returns:
            Antecedent: Antecedent of the rule
        """
        return self._antecedent

    cpdef Consequent get_consequent(self):
        """Get the consequent
        
        Returns:
           Consequent: Consequent of the rule
        """
        return self._consequent

    cpdef void set_consequent(self, Consequent consequent):
        """Set the consequent

        Args:
           consequent (Consequent): New consequent
        """
        self._consequent = consequent

    cdef double[:] get_membership_values(self, double[:] attribute_vector):
        """Get the membership values array for the antecedent with the given attribute vector
        
        Args:
            attribute_vector (double[]): Input vector whose membership values are computed

        Returns:
            double[]: Membership values
        """
        return self._antecedent.get_membership_values(attribute_vector)

    cdef double get_compatible_grade_value(self, double[:] attribute_vector):
        """Get the compatible grade value for the antecedent with the given attribute vector
        
        Args:
            attribute_vector (double[]): Input vector whose compatible grade value is computed

        Returns:
            double: Compatible grade value
        """
        return self._antecedent.get_compatible_grade_value(attribute_vector)

    cpdef AbstractClassLabel get_class_label(self):
        """Get the class label of the consequent
        
        Returns:
            AbstractClassLabel: Class label
        """
        return self._consequent.get_class_label()


    cpdef bint is_rejected_class_label(self):
        """Check if the class label is rejected
        
        Returns:
            bool: True if it is rejected and false otherwise
        """
        return self._consequent.get_class_label().is_rejected()

    cdef AbstractRuleWeight get_rule_weight(self):
        """Get the rule weight object. Can only be accessed from Cython
        
        Returns:
            AbstractRuleWeight: Rule weight object
        """
        return self._consequent.get_rule_weight()

    cpdef int get_length(self):
        """Get the length of the antecedent (Number of not DC fuzzy sets inside)
        
        Returns:
            int: Length of the antecedent
        """
        return self.get_antecedent().get_length()

    cpdef int get_antecedent_array_size(self):
        """Get the length of the antecedent indices array
        
        Returns:
            int: Length of the antecedent indices array
        """
        return self.get_antecedent().get_array_size()

    cpdef double get_fitness_value(self, double[:] attribute_vector):
        """Get the fitness value of the rule for the given input vector
        
        Args:
            attribute_vector (double[]): Input vector 

        Returns:
            double: Fitness value
        """
        Exception("AbstractRule is abstract")

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, AbstractRule):
            return False

        return self._antecedent == other.get_antecedent() and self._consequent == other.get_consequent()

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"Antecedent: {self._antecedent} => Consequent: {self._consequent}"

    cpdef str get_linguistic_representation(self):
        """Get the linguistic representation of the rule (IF .... THEN ...)
        
        Returns:
            str: Linguistic representation
        """
        return f"IF\t{self._antecedent.get_linguistic_representation()} THEN {self._consequent.get_linguistic_representation()} RW: {self.get_rule_weight()}"

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("rule")

        root.append(self._antecedent.to_xml())
        root.append(self._consequent.to_xml())

        return root

    cpdef Knowledge get_knowledge(self):
        """Get the knowledge base
        
        Returns:
            Knowledge: Knowledge base
        """
        return self._antecedent.get_knowledge()

    cpdef FuzzySet get_fuzzy_set_object(self, int dim_index):
        """Get the fuzzy set at the given dimension in the knowledge base of this antecedent
        
        Args:
            dim_index (int): Index of the dimension where the fuzzy set is fetched 

        Returns:
            FuzzySet: Fuzzy set fetched
        """
        fuzzy_set_index = self.get_antecedent().get_antecedent_indices()[dim_index]
        return self.get_knowledge().get_fuzzy_set(dim_index, fuzzy_set_index)

    cpdef str get_var_name(self, int dim_index):
        """Get the name of the variable at the given dimension
        Args:
            dim_index (int ): Index of the dimension where the variable name is fetched

        Returns:
            str: Variable name
        """
        return self.get_knowledge().get_fuzzy_variable(dim_index).get_name()