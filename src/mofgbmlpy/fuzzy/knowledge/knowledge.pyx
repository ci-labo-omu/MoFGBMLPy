import xml.etree.cElementTree as xml_tree
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import cython

from mofgbmlpy.exception.uninitialized_knowledge_exception import UninitializedKnowledgeException
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable


cdef class Knowledge:
    """Knowledge base of a fuzzy system

    Attributes:
        __fuzzy_vars (FuzzyVariable[]): Fuzzy variables in this knowledge
    """
    def __cinit__(self, FuzzyVariable[:] fuzzy_vars=None, bint do_init=True):
        """Constructor

        Args:
            fuzzy_vars (FuzzyVariable[]): Fuzzy variables in this knowledge
        """
        if not do_init:
            self.ptr = NULL
            return

        cdef vector[FuzzyVariableCpp*] cpp_fuzzy_vars
        if fuzzy_vars is not None and fuzzy_vars.shape[0] != 0:
            cpp_fuzzy_vars.reserve(fuzzy_vars.shape[0])
            for i in range(fuzzy_vars.shape[0]):
                cpp_fuzzy_vars.push_back(fuzzy_vars[i].ptr.clone())

        self.ptr = new KnowledgeCpp(cpp_fuzzy_vars)

    def __dealloc__(self):
        """Destructor"""
        if self.ptr != NULL:
            del self.ptr


    cpdef FuzzyVariable get_fuzzy_variable(self, int dim):
        """Get the fuzzy variable at the given index
        
        Args:
            dim (int): Dimension of the fetched fuzzy variable

        Returns:
            FuzzyVariable: Fetched variable
        """
        return FuzzyVariable.wrap(self.ptr.get_fuzzy_variable(dim))

    cpdef FuzzySet get_fuzzy_set(self, int dim, int fuzzy_set_index):
        """Get the fuzzy set of the given dimension at a given index
        
        Args:
            dim (int): Dimension of the fuzzy variable in which this fuzzy set is
            fuzzy_set_index (int): Index of the fuzzy set in the fuzzy variable

        Returns:
            FuzzySet: Fuzzy set fetched
        """
        return FuzzySet.wrap(self.ptr.get_fuzzy_set(dim, fuzzy_set_index))

    cpdef int get_num_fuzzy_sets(self, int dim):
        """Get the number of fuzzy sets in the fuzzy variable at the given dimension
        
        Args:
            dim (int): Dimension where the fuzzy variable whose number of fuzzy sets is fetched 

        Returns:
            int: Number of fuzzy sets
        """
        return self.ptr.get_num_fuzzy_sets(dim)

    cpdef void set_fuzzy_vars(self, FuzzyVariable[:] fuzzy_vars):
        """Set the list of fuzzy variables of this knowledge base
        
        Args:
            fuzzy_vars (FuzzyVariable[]): Array of the new fuzzy variables
        """
        cdef vector[FuzzyVariableCpp*] cpp_fuzzy_vars
        if fuzzy_vars is not None and fuzzy_vars.shape[0] != 0:
            cpp_fuzzy_vars.reserve(fuzzy_vars.shape[0])
            for i in range(fuzzy_vars.shape[0]):
                cpp_fuzzy_vars.push_back(fuzzy_vars[i].ptr.clone())
        self.ptr.set_fuzzy_vars(cpp_fuzzy_vars)

    cpdef FuzzyVariable[:] get_fuzzy_vars(self):
        """Get the list of all fuzzy variables
        
        Returns:
            FuzzyVariable[]: Fuzzy variables of this knowledge base
        """
        cdef vector[FuzzyVariableCpp*] cpp_fuzzy_vars = self.ptr.get_fuzzy_vars()
        cdef FuzzyVariable[:] fuzzy_vars = np.empty(cpp_fuzzy_vars.size(), dtype=FuzzyVariable)
        cdef FuzzyVariable fv
        cdef int i
        for i in range(cpp_fuzzy_vars.size()):
            fv = FuzzyVariable.wrap(cpp_fuzzy_vars[i])
            fuzzy_vars[i] = fv

        return fuzzy_vars

    cpdef double get_membership_value_py(self, double attribute_value, int dim, int fuzzy_set_index):
        """Get the membership value of the given attribute value with the fuzzy set at the given dimension and given index
        
        Args:
            attribute_value (double): Attribute value whose compatibility is computed 
            dim (int): Dimension index where there is the fuzzy variable where the fuzzy set is 
            fuzzy_set_index (int): Index of the fuzzy set in the fuzzy variable 

        Returns:
            double: Membership value
        """
        return self.ptr.get_membership_value(attribute_value, dim, fuzzy_set_index)

    @cython.boundscheck(True)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_index):
        """Get the membership value of the given attribute value with the fuzzy set at the given dimension and given index (Can only be accessed from Cython code)
        
        Args:
            attribute_value (double): Attribute value whose compatibility is computed 
            dim (int): Dimension index where there is the fuzzy variable where the fuzzy set is 
            fuzzy_set_index (int): Index of the fuzzy set in the fuzzy variable 

        Returns:
            double: Membership value
        """
        return self.ptr.get_membership_value(attribute_value, dim, fuzzy_set_index)

    cpdef int get_num_dim(self):
        """Get the number of dimensions of this knowledge base
        
        Returns:
            int: Number of dimensions
        """
        return self.ptr.get_num_dim()

    cpdef double get_support(self, int dim, int fuzzy_set_index):
        """Get the support value associated to the membership function of the fuzzy set at the given index in the variable at the given dimension: area covered by this function in the space "variable_domain x [0, 1]"
        Args:
            dim (int): Dimension index where there is the fuzzy variable where the fuzzy set is 
            fuzzy_set_index (int): Index of the fuzzy set in the fuzzy variable 
        Returns:
            double: Support value
        """
        return self.ptr.get_support(dim, fuzzy_set_index)

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return self.ptr.to_string().decode('utf-8')

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = Knowledge.wrap(self.ptr)
        memo[id(self)] = new_object
        return new_object

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, Knowledge):
            return False

        cdef Knowledge other_c = <Knowledge> other
        return self.ptr[0] == other_c.ptr[0]

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


    def plot_fuzzy_variables(self):
        """Plot all the fuzzy variables of this knowledge base (one plot per variable)"""
        for i in range(self.get_num_dim()):
            ax = plt.axes()
            ax = self.get_fuzzy_variable(i).get_plot(ax)
            plt.show()

    @staticmethod
    cdef Knowledge wrap(KnowledgeCpp * wrapped_ptr):
        if wrapped_ptr == NULL:
            raise ValueError("pointer is NULL")
        cdef Knowledge new_object = Knowledge(do_init=False)
        new_object.ptr = wrapped_ptr.clone()
        return new_object