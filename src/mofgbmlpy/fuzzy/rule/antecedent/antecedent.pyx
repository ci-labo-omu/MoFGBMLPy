import xml.etree.cElementTree as xml_tree
import copy

import numpy as np
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
cimport cython
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport round

cdef class Antecedent:
    # cdef public object __antecedent_indices
    # cdef public Knowledge __knowledge

    def __init__(self, int[:] antecedent_indices, Knowledge knowledge):
        self.__antecedent_indices = antecedent_indices
        self.__knowledge = knowledge

    cpdef int get_array_size(self):
        return self.__antecedent_indices.shape[0]

    cpdef int[:] get_antecedent_indices(self):
        return self.__antecedent_indices

    cpdef void set_antecedent_indices(self, int[:] new_indices):
        self.__antecedent_indices = new_indices

    cpdef double[:] get_compatible_grade(self, double[:] attribute_vector):
        cdef int i
        cdef int size = self.get_array_size()
        cdef double[:] grade = np.zeros(size, dtype=np.float64)
        cdef int[:] antecedent_indices = self.__antecedent_indices

        if size != attribute_vector.shape[0]:
            # with cython.gil:
            raise ValueError("antecedent_indices and attribute_vector must have the same length")

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
                # with cython.gil:
                raise ValueError("Illegal argument")

        return grade

    cdef double get_compatible_grade_value(self, double[:] attribute_vector):
        cdef int i
        cdef int size = self.get_array_size()
        cdef double grade_value = 1
        cdef double val
        cdef int[:] antecedent_indices = self.__antecedent_indices

        if size != attribute_vector.shape[0]:
            # with cython.gil:
            raise ValueError("antecedent_indices and attribute_vector must have the same length")

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
                raise Exception("Invalid antecedent_indices")

        return grade_value

    cpdef double get_compatible_grade_value_py(self, double[:] attribute_vector):
        cdef double compatible_grade_value = self.get_compatible_grade_value(attribute_vector)
        if compatible_grade_value == -1:
            # Error code
            # with cython.gil:
            raise ValueError("Illegal argument")

    cpdef int get_length(self):
        return np.count_nonzero(self.__antecedent_indices)

    def __deepcopy__(self, memo={}):
        cdef int[:] antecedent_indices_copy = np.empty(self.get_array_size(), dtype=int)
        cdef int i

        for i in range(antecedent_indices_copy.shape[0]):
            antecedent_indices_copy[i] = self.__antecedent_indices[i]

        new_antecedent = Antecedent(antecedent_indices_copy, knowledge=self.__knowledge)
        memo[id(self)] = new_antecedent
        return new_antecedent

    def __eq__(self, other):
        return np.array_equal(self.__antecedent_indices, other.get_antecedent_indices()) and self.__knowledge == other.__knowledge

    def __repr__(self):
        txt = "["
        for i in range(self.__antecedent_indices.shape[0]):
            txt += f"{self.__antecedent_indices[i]} "
        return txt + "]"

    def to_xml(self):
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
        return self.__knowledge