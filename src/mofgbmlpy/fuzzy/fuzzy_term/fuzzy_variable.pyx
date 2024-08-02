import xml.etree.cElementTree as xml_tree
import copy

import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet


cdef class FuzzyVariable:
    def __init__(self, FuzzySet[:] fuzzy_sets, double[:] support_values, str name="unnamed_var", domain=None):
        if name is None:
            raise Exception("name can't be done")
        if fuzzy_sets is None or len(fuzzy_sets) == 0 or support_values is None or len(support_values) != len(fuzzy_sets):
            raise Exception("fuzzy_sets and support_values must have the same length and at least one element")

        self.__support_values = support_values
        self.__fuzzy_sets = fuzzy_sets
        self.__name = name

        if domain is None:
            self.__domain = np.array([0.0, 1.0])
        else:
            if len(domain) != 2:
                raise Exception("domain must be an array of double size 2 (min, max)")
            elif domain[0] > domain[1]:
                raise Exception("domain's first value must be lesser than the second one")
            self.__domain = domain

    cpdef str get_name(self):
        return self.__name

    cdef double get_membership_value(self, int fuzzy_set_index, double x):
        if fuzzy_set_index >= self.__fuzzy_sets.shape[0]:
            raise Exception(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")
        cdef FuzzySet fuzzy_set = self.__fuzzy_sets[fuzzy_set_index]
        return fuzzy_set.get_membership_value(x)

    def get_membership_value_py(self, int fuzzy_set_index, double x):
        self.get_membership_value(fuzzy_set_index, x)

    cpdef int get_length(self):
        return len(self.__fuzzy_sets)

    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index):
        if fuzzy_set_index >= self.__fuzzy_sets.shape[0]:
            raise Exception(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")
        return self.__fuzzy_sets[fuzzy_set_index]

    cpdef double get_support(self, int fuzzy_set_index):
        if fuzzy_set_index >= self.__support_values.shape[0]:
            raise Exception(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")
        return self.__support_values[fuzzy_set_index]

    cpdef get_fuzzy_sets(self):
        return self.__fuzzy_sets

    cpdef get_support_values(self):
        return self.__support_values

    cpdef get_domain(self):
        return self.__domain

    def get_plot(self, ax):
        cdef int i
        cdef FuzzySet fuzzy_set
        cdef cnp.ndarray[double, ndim=2] points

        ax.set_title(self.get_name())
        for i in range(self.get_length()):
            fuzzy_set = self.get_fuzzy_set(i)
            points = fuzzy_set.get_function().get_plot_points()
            ax.plot(points[:,0], points[:,1])

        return ax

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        txt = f"Fuzzy variable for {self.__name}:\n"
        cdef int i
        for i in range(len(self.__fuzzy_sets)):
            txt += f"\t{self.__fuzzy_sets[i]}\n"
        return txt

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            (object) Deep copy of this object
        """
        cdef double[:] support_values_copy = np.copy(self.__support_values)
        cdef FuzzySet[:] fuzzy_sets_copy = np.empty(self.__fuzzy_sets.shape[0], dtype=object)
        cdef int i

        for i in range(fuzzy_sets_copy.shape[0]):
            fuzzy_sets_copy[i] = copy.deepcopy(self.__fuzzy_sets[i])

        cdef FuzzyVariable new_object = FuzzyVariable(fuzzy_sets_copy, support_values_copy, self.__name, np.copy(self.__domain))
        memo[id(self)] = new_object
        return new_object

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("fuzzySets")

        for i in range(self.get_length()):
            root.append(self.get_fuzzy_set(i).to_xml())

        return root


    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, FuzzyVariable):
            return False


        return (np.array_equal(self.__fuzzy_sets, other.get_fuzzy_sets()) and
                np.array_equal(self.__domain, other.get_domain()) and
                np.array_equal(self.__support_values, other.get_support_values()) and
                self.__name == other.get_name())