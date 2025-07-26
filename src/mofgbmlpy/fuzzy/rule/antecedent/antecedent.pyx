import xml.etree.cElementTree as xml_tree
import copy

import numpy as np

from mofgbmlpy.exception.incompatible_antecedent_index_with_input import IncompatibleAntecedentIndexWithInput
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
cimport cython
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet, FuzzySetCpp
import matplotlib.pyplot as plt


cdef class Antecedent:
    """Antecedent part of fuzzy rules

    Attributes:
        __antecedent_indices (int[]): Indices of the fuzzy sets of this antecedent
        __knowledge (Knowledge): Knowledge base
    """
    def __cinit__(self, int[:] antecedent_indices, Knowledge knowledge, bint do_init=True):
        """Constructor

        Args:
            antecedent_indices (int[]): Indices of the fuzzy sets of this antecedent
            knowledge (Knowledge): Knowledge base
        """

        if not do_init:
            self.ptr = NULL
            return

        if antecedent_indices is None or knowledge is None:
            raise TypeError("Parameters can't be None")

        cdef vector[int] cpp_antecedent_indices
        if antecedent_indices is not None and antecedent_indices.shape[0] != 0:
            cpp_antecedent_indices.reserve(antecedent_indices.shape[0])
            for i in range(antecedent_indices.shape[0]):
                cpp_antecedent_indices.push_back(antecedent_indices[i])

        self.ptr = new AntecedentCpp(cpp_antecedent_indices, knowledge.ptr.clone())

    def __dealloc__(self):
        """Destructor"""
        if self.ptr != NULL:
            del self.ptr

    cpdef int get_array_size(self):
        """Get the size of the antecedent array (number of dimensions)
        
        Returns:
            int: Antecedent array size
        """
        return self.ptr.get_array_size()

    cpdef int[:] get_antecedent_indices(self):
        """Get the antecedent indices
        
        Returns:
            int[]: Antecedent indices
        """
        return np.array(self.ptr.get_antecedent_indices(), dtype=np.int32)

    cpdef void set_antecedent_indices(self, int[:] new_indices):
        """Set the antecedent indices
        
        Args:
            new_indices (int[]): New indices for this antecedent
        """
        if new_indices is None:
            raise TypeError("New antecedent indices can't be None")

        cdef vector[int] cpp_antecedent_indices

        if new_indices.shape[0] != 0:
            cpp_antecedent_indices.reserve(new_indices.shape[0])
            for i in range(new_indices.shape[0]):
                cpp_antecedent_indices.push_back(new_indices[i])
        self.ptr.set_antecedent_indices(cpp_antecedent_indices)

    cpdef double[:] get_membership_values(self, double[:] attribute_vector):
        """Get the membership values of the given attribute vector with this antecedent for each dimension
        
        Args:
            attribute_vector (double[]): Attribute vector whose membership values are computed 

        Returns:
            double[]: Membership value for each dimension
        """
        cdef vector[double] cpp_attribute_vector
        if attribute_vector is not None and attribute_vector.shape[0] != 0:
            cpp_attribute_vector.reserve(attribute_vector.shape[0])
            for i in range(attribute_vector.shape[0]):
                cpp_attribute_vector.push_back(attribute_vector[i])

        return np.array(self.ptr.get_membership_values(cpp_attribute_vector), dtype=np.float64)

    cdef double get_compatible_grade_value(self, double[:] attribute_vector):
        """Get the compatibility grade of the given attribute vector with this antecedent. Can only be accesses from Cython code

        Args:
            attribute_vector (double[]): Attribute vector whose compatibility is computed 

        Returns:
            double[]: Compatibility grade
        """
        cdef vector[double] cpp_attribute_vector
        if attribute_vector is not None and attribute_vector.shape[0] != 0:
            cpp_attribute_vector.reserve(attribute_vector.shape[0])
            for i in range(attribute_vector.shape[0]):
                cpp_attribute_vector.push_back(attribute_vector[i])

        return self.ptr.get_compatible_grade_value(cpp_attribute_vector)

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
        return self.ptr.get_length()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = Antecedent.wrap(self.ptr)
        memo[id(self)] = new_object
        return new_object

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, Antecedent):
            return False

        cdef Antecedent other_antecedent = <Antecedent>other
        return self.ptr[0] == other_antecedent.ptr[0]

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return self.ptr.to_string().decode('utf-8')

    cpdef str get_linguistic_representation(self):
        """Get the linguistic representation of the antecedent (... IS ... AND ... IS ...)
        
        Returns:
            str: Linguistic representation of the antecedent
        """
        return self.ptr.get_linguistic_representation().decode('utf-8')

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("antecedent")
        # for dim_i in range(len(self.__antecedent_indices)):
        #     root.append(self.__knowledge.get_fuzzy_set(dim_i, self.__antecedent_indices[dim_i]).to_xml())

        cdef int[:] cpp_antecedent_indices = self.get_antecedent_indices()

        fuzzy_set_list = xml_tree.SubElement(root, "fuzzySetList")
        for dim_i in range(len(cpp_antecedent_indices)):
            fuzzy_set_id = xml_tree.SubElement(fuzzy_set_list, "fuzzySetID")
            fuzzy_set_id.set("dimension", str(dim_i))
            fuzzy_set_id.text = str(cpp_antecedent_indices[dim_i])
        return root

    cpdef get_knowledge(self):
        """Get the knowledge base
        
        Returns:
            Knowledge: Knowledge base
        """
        return Knowledge.wrap(self.ptr.get_knowledge())

    cpdef set_knowledge(self, Knowledge new_knowledge):
        """Set the knowledge base
        
        Args:
            new_knowledge (Knowledge): New knowledge base
        """
        if new_knowledge is None:
            raise TypeError("New knowledge base can't be None")

        cdef KnowledgeCpp* new_knowledge_ptr = new_knowledge.ptr.clone()
        self.ptr.set_knowledge(new_knowledge_ptr)

    def get_plot(self, ax, int dim):
        """Draw the antecedent fuzzy sets on the given matplotlib Axes object

        Args:
            ax (matplotlib.axes.Axes): Axes object
            dim (int): Dimension to plot

        Returns:
            matplotlib.axes.Axes: The axes object where we drew
        """
        cdef FuzzySetCpp* fuzzy_set
        cdef cnp.ndarray[double, ndim=2] points

        fuzzy_set = self.ptr.get_knowledge().get_fuzzy_set(dim, self.get_antecedent_indices()[dim])

        points = np.array(fuzzy_set.get_function().get_plot_points(0, 1))
        ax.plot(points[:,0], points[:,1])
        ax.set_title(f"x_{dim}")
        ax.set_xlim([0,1])
        ax.set_ylim([0,1.1])

        return ax

    def plot_antecedent(self):
        fig, axes = plt.subplots(1, self.get_array_size(), figsize=(25, 3))

        for i in range(self.get_array_size()):
            axes[i] = self.get_plot(axes[i], i)

        plt.show()

    @staticmethod
    cdef Antecedent wrap(AntecedentCpp * wrapped_ptr):
        if wrapped_ptr == NULL:
            raise ValueError("pointer is NULL")
        cdef Antecedent new_object = Antecedent(None, None, do_init=False)
        new_object.ptr = wrapped_ptr.clone()
        return new_object
