import xml.etree.cElementTree as xml_tree
from copy import deepcopy
import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable

cdef class Knowledge:
    def __init__(self, fuzzy_sets=None):
        if fuzzy_sets is None:
            self.__fuzzy_sets = np.empty(0, dtype=object)
        else:
            self.__fuzzy_sets = fuzzy_sets

    cpdef FuzzyVariable get_fuzzy_variable(self, int dim):
        return self.__fuzzy_sets[dim]

    cpdef FuzzySet get_fuzzy_set(self, int dim, int fuzzy_set_id):
        cdef FuzzyVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.shape[0] == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        cdef FuzzyVariable var = fuzzy_sets[dim]
        return var.get_fuzzy_set(fuzzy_set_id)

    cpdef int get_num_fuzzy_sets(self, int dim):
        cdef FuzzyVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.shape[0] == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        cdef FuzzyVariable var = fuzzy_sets[dim]
        return var.get_length()

    cpdef void set_fuzzy_sets(self, FuzzyVariable[:] fuzzy_sets):
        # cdef FuzzyVariable[:] self_fuzzy_sets = self.__fuzzy_sets
        #     if self_fuzzy_sets is None or self_fuzzy_sets.shape[0] == 0:
        if self.__fuzzy_sets is not None and len(self.__fuzzy_sets) != 0:
            # with cython.gil:
            raise Exception("You can't overwrite fuzzy sets. You must call clear before doing so")

        self.__fuzzy_sets = fuzzy_sets

    cpdef FuzzyVariable[:] get_fuzzy_sets(self):
        cdef FuzzyVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.shape[0] == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        return fuzzy_sets

    cpdef double get_membership_value_py(self, double attribute_value, int dim, int fuzzy_set_id):
        cdef FuzzyVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.shape[0] == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")
        cdef FuzzyVariable var = fuzzy_sets[dim]
        return var.get_membership_value(fuzzy_set_id, attribute_value)

    cdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_id):
        cdef FuzzyVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.shape[0] == 0:
            return -1
        cdef FuzzyVariable var = fuzzy_sets[dim]
        return var.get_membership_value(fuzzy_set_id, attribute_value)

    cpdef int get_num_dim(self):
        cdef FuzzyVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None:
            return 0
        return fuzzy_sets.shape[0]

    cpdef void clear(self):
        self.__fuzzy_sets = np.empty(0, dtype=object)

    cpdef double get_support(self, int  dim, int  fuzzy_set_id):
        return self.get_fuzzy_variable(dim).get_support(fuzzy_set_id)

    def __repr__(self):
        txt = ""
        for i in range(self.get_num_dim()):
            txt = f"{txt}{str(self.__fuzzy_sets[i])}\n"
        return txt

    def __deepcopy__(self, memo={}):
        cdef FuzzyVariable[:] fuzzy_sets = self.__fuzzy_sets
        cdef FuzzyVariable[:] fuzzy_sets_copy = np.empty(fuzzy_sets.shape[0], dtype=object)
        cdef int i

        for i in range(fuzzy_sets.shape[0]):
            fuzzy_sets_copy[i] = deepcopy(fuzzy_sets[i])

        new_knowledge = Knowledge(fuzzy_sets_copy)
        memo[id(self)] = new_knowledge

        return new_knowledge

    def to_xml(self):
        root = xml_tree.Element("knowledgeBase")
        for i in range(self.get_num_dim()):
            var = self.get_fuzzy_variable(i).to_xml()
            var.set("dimension", str(i))
            root.append(var)

        return root

