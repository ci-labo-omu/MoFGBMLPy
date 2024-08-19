import xml.etree.cElementTree as xml_tree
import copy

import numpy as np

from mofgbmlpy.exception.incompatible_antecedent_index_with_input import IncompatibleAntecedentIndexWithInput
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
cimport cython
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport round

cdef class Antecedent:
    """Antecedent part of fuzzy rules

    Attributes:
        __antecedent_indices (int[]): Indices of the fuzzy sets of this antecedent
        __knowledge (Knowledge): Knowledge base
    """
    def __init__(self, int[:] antecedent_indices, Knowledge knowledge):
        """Constructor

        Args:
            antecedent_indices (int[]): Indices of the fuzzy sets of this antecedent
            knowledge (Knowledge): Knowledge base
        """
        if antecedent_indices is None or knowledge is None:
            raise TypeError("Parameters can't be None")

        self.__antecedent_indices = antecedent_indices
        self.__knowledge = knowledge

    cpdef int get_array_size(self):
        """Get the size of the antecedent array (number of dimensions)
        
        Returns:
            int: Antecedent array size
        """
        return self.__antecedent_indices.shape[0]

    cpdef int[:] get_antecedent_indices(self):
        """Get the antecedent indices
        
        Returns:
            int[]: Antecedent indices
        """
        return self.__antecedent_indices

    cpdef void set_antecedent_indices(self, int[:] new_indices):
        """Set the antecedent indices
        
        Args:
            new_indices (int[]): New indices for this antecedent
        """
        if new_indices is None:
            raise TypeError("new_indices can't be None")
        self.__antecedent_indices = new_indices

    cpdef double[:] get_membership_values(self, double[:] attribute_vector):
        """Get the membership values of the given attribute vector with this antecedent for each dimension
        
        Args:
            attribute_vector (double[]): Attribute vector whose membership values are computed 

        Returns:
            double[]: Membership value for each dimension
        """
        cdef int i
        cdef int size = self.get_array_size()
        cdef double[:] grade = np.zeros(size, dtype=np.float64)
        cdef int[:] antecedent_indices = self.__antecedent_indices

        if attribute_vector is None :
            raise TypeError("antecedent_indices must not be None")
        elif size != attribute_vector.shape[0]:
            raise ValueError("antecedent_indices must have the same length as attribute_vector")

        if size > self.__knowledge.get_num_dim():
            raise IndexError("The given number of dimensions is out of bounds for the current knowledge")

        for i in range(size):
            val = attribute_vector[i]
            if antecedent_indices[i] < 0 and val < 0:
                # categorical
                grade[i] = 1.0 if antecedent_indices[i] == round(val) else 0.0
            elif antecedent_indices[i] > 0 and val >= 0:
                # numerical
                grade[i] = self.__knowledge.get_membership_value(val, i, antecedent_indices[i])
            elif antecedent_indices[i] == 0:
                # don't care
                grade[i] = 1.0
            else:
                raise IncompatibleAntecedentIndexWithInput(i, val, antecedent_indices[i])

        return grade

    cdef double get_compatible_grade_value(self, double[:] attribute_vector):
        """Get the compatibility grade of the given attribute vector with this antecedent. Can only be accesses from Cython code

        Args:
            attribute_vector (double[]): Attribute vector whose compatibility is computed 

        Returns:
            double[]: Compatibility grade
        """
        cdef int i
        cdef int size = self.get_array_size()
        cdef double grade_value = 1
        cdef double val
        cdef int[:] antecedent_indices = self.__antecedent_indices

        if size != attribute_vector.shape[0]:
            # with cython.gil:
            raise ValueError("antecedent_indices and attribute_vector must have the same length")

        if size > self.__knowledge.get_num_dim():
            raise IndexError("The given number of dimensions is out of bounds for the current knowledge")

        for i in range(size):
        # for i in prange(size, nogil=True):
            val = attribute_vector[i]
            if antecedent_indices[i] < 0 and val < 0:
                # categorical
                if antecedent_indices[i] != round(val):
                    grade_value = 0.0
            elif antecedent_indices[i] > 0 and val >= 0:
                # numerical
                grade_value *= self.__knowledge.get_membership_value(val, i, antecedent_indices[i])
            elif antecedent_indices[i] == 0:
                continue
            else:
                raise IncompatibleAntecedentIndexWithInput(i, val, antecedent_indices[i])

        return grade_value

    def get_compatible_grade_value_py(self, double[:] attribute_vector):
        """Get the compatibility grade of the given attribute vector with this antecedent

        Args:
            attribute_vector (double[]): Attribute vector whose compatibility is computed

        Returns:
            double[]: Compatibility grade
        """
        return self.get_compatible_grade_value(attribute_vector)

    cpdef int get_length(self):
        """Get the length of the antecedent
        
        Returns:
            int: Number of antecedent indices that do not correspond to don't care (i.e. number of non-null indices)
        """
        return np.count_nonzero(self.__antecedent_indices)

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef int[:] antecedent_indices_copy = np.empty(self.get_array_size(), dtype=int)
        cdef int i

        for i in range(antecedent_indices_copy.shape[0]):
            antecedent_indices_copy[i] = self.__antecedent_indices[i]

        new_antecedent = Antecedent(antecedent_indices_copy, knowledge=self.__knowledge)
        memo[id(self)] = new_antecedent
        return new_antecedent

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, Antecedent):
            return False

        return np.array_equal(self.__antecedent_indices, other.get_antecedent_indices()) and self.__knowledge == other.get_knowledge()

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        txt = "["
        for i in range(self.__antecedent_indices.shape[0]):
            txt += f"{self.__antecedent_indices[i]} "
        return txt + "]"

    cpdef str get_linguistic_representation(self):
        """Get the linguistic representation of the antecedent (... IS ... AND ... IS ...)
        
        Returns:
            str: Linguistic representation of the antecedent
        """
        txt = ""
        cdef int i
        cdef FuzzyVariable var

        for i in range(self.get_array_size()):
            var = self.__knowledge.get_fuzzy_variable(i)

            if self.__antecedent_indices[i] != 0:
                txt = f" {txt} {var.get_name()} IS {var.get_fuzzy_set(self.__antecedent_indices[i]).get_term()} AND"

        if txt == "":
            txt = "[don't care]"

        return txt[:-4] # Remove the last "AND "

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("antecedent")
        # for dim_i in range(len(self.__antecedent_indices)):
        #     root.append(self.__knowledge.get_fuzzy_set(dim_i, self.__antecedent_indices[dim_i]).to_xml())

        fuzzy_set_list = xml_tree.SubElement(root, "fuzzySetList")
        for dim_i in range(len(self.__antecedent_indices)):
            fuzzy_set_id = xml_tree.SubElement(fuzzy_set_list, "fuzzySetID")
            fuzzy_set_id.set("dimension", str(dim_i))
            fuzzy_set_id.text = str(self.__antecedent_indices[dim_i])
        return root

    cpdef get_knowledge(self):
        """Get the knowledge base
        
        Returns:
            Knowledge: Knowledge base
        """
        return self.__knowledge

    cpdef set_knowledge(self, Knowledge new_knowledge):
        """Set the knowledge base
        
        Args:
            new_knowledge (Knowledge): New knowledge base
        """
        self.__knowledge = new_knowledge