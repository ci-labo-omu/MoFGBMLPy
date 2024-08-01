import xml.etree.cElementTree as xml_tree
from copy import deepcopy
import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable


cdef class Knowledge:
    def __init__(self, fuzzy_vars=None):
        if fuzzy_vars is None:
            self.__fuzzy_vars = np.empty(0, dtype=object)
        else:
            self.__fuzzy_vars = fuzzy_vars

    cpdef FuzzyVariable get_fuzzy_variable(self, int dim):
        return self.__fuzzy_vars[dim]

    cpdef FuzzySet get_fuzzy_set(self, int dim, int fuzzy_set_index):
        cdef FuzzyVariable[:] fuzzy_vars = self.__fuzzy_vars
        if fuzzy_vars.shape[0] == 0:
            raise Exception("Knowledge is not yet initialized (no fuzzy set)")

        cdef FuzzyVariable var = fuzzy_vars[dim]
        return var.get_fuzzy_set(fuzzy_set_index)

    cpdef int get_num_fuzzy_sets(self, int dim):
        cdef FuzzyVariable[:] fuzzy_vars = self.__fuzzy_vars
        if fuzzy_vars.shape[0] == 0:
            raise Exception("Knowledge is not yet initialized (no fuzzy set)")

        cdef FuzzyVariable var = fuzzy_vars[dim]
        return var.get_length()

    cpdef void set_fuzzy_vars(self, FuzzyVariable[:] fuzzy_vars):
        if fuzzy_vars is None:
            self.__fuzzy_vars = np.empty(0, object)
        else:
            self.__fuzzy_vars = fuzzy_vars

    cpdef FuzzyVariable[:] get_fuzzy_vars(self):
        return self.__fuzzy_vars

    cpdef double get_membership_value_py(self, double attribute_value, int dim, int fuzzy_set_index):
        return self.get_membership_value(attribute_value, dim, fuzzy_set_index)

    cdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_index):
        cdef FuzzyVariable[:] fuzzy_vars = self.__fuzzy_vars
        if fuzzy_vars.shape[0] == 0:
            raise Exception("Knowledge is not yet initialized (no fuzzy set)")
        if dim < 0 or dim >= fuzzy_vars.shape[0]:
            raise Exception("The dim index is out of bounds for the current knowledge")

        cdef FuzzyVariable var = fuzzy_vars[dim]
        return var.get_membership_value(fuzzy_set_index, attribute_value)

    cpdef int get_num_dim(self):
        cdef FuzzyVariable[:] fuzzy_sets = self.__fuzzy_vars
        return fuzzy_sets.shape[0]

    cpdef double get_support(self, int dim, int fuzzy_set_index):
        return self.get_fuzzy_variable(dim).get_support(fuzzy_set_index)

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        txt = ""
        for i in range(self.get_num_dim()):
            txt = f"{txt}{str(self.__fuzzy_vars[i])}\n"
        return txt

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            (object) Deep copy of this object
        """
        cdef FuzzyVariable[:] fuzzy_vars = self.__fuzzy_vars
        cdef FuzzyVariable[:] fuzzy_vars_copy = np.empty(fuzzy_vars.shape[0], dtype=object)
        cdef int i

        for i in range(fuzzy_vars.shape[0]):
            fuzzy_vars_copy[i] = deepcopy(fuzzy_vars[i])

        new_knowledge = Knowledge(fuzzy_vars_copy)
        memo[id(self)] = new_knowledge

        return new_knowledge

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, Knowledge):
            return False
        return np.array_equal(self.__fuzzy_vars, other.get_fuzzy_vars())

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("knowledgeBase")
        for i in range(self.get_num_dim()):
            var = self.get_fuzzy_variable(i).to_xml()
            var.set("dimension", str(i))
            root.append(var)

        return root

