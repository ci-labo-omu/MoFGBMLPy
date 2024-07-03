import copy

import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set cimport FuzzySet


cdef class LinguisticVariable:
    def __init__(self, fuzzy_sets=None, name=None, support_values=None):
        if fuzzy_sets is None or len(fuzzy_sets) == 0:
            fuzzy_sets = []
            support_values = []
        elif support_values is None:
            # with cython.gil:
            raise TypeError('Support value cannot be None if fuzzy_sets is not None and not empty')

        self.__support_values = support_values
        self.__fuzzy_sets = fuzzy_sets
        self.__concept = str(name)
        self.__domain = np.array([0.0, 1.0])

    cpdef str get_concept(self):
        return self.__concept

    cdef double get_membership_value(self, int fuzzy_set_index, double x):
        if fuzzy_set_index > self.__fuzzy_sets.shape[0]:
            # with cython.gil:
            raise Exception(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")
        cdef FuzzySet fuzzy_set = self.__fuzzy_sets[fuzzy_set_index]
        return fuzzy_set.get_membership_value(x)

    cpdef int get_length(self):
        return len(self.__fuzzy_sets)

    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index):
        return self.__fuzzy_sets[fuzzy_set_index]

    cpdef double get_support(self, int fuzzy_set_id):
        return self.__support_values[fuzzy_set_id]

    cpdef get_fuzzy_sets(self):
        return self.__fuzzy_sets

    cpdef get_domain(self):
        return self.__domain

    def __repr__(self):
        txt = f"Fuzzy variable for {self.__concept}:\n"
        cdef int i
        for i in range(len(self.__fuzzy_sets)):
            txt += f"\t{self.__fuzzy_sets[i]}\n"
        return txt

    def __deepcopy__(self, memo={}):
        cdef double[:] support_values_copy = np.copy(self.__support_values)
        cdef FuzzySet[:] fuzzy_sets_copy = np.empty(self.__fuzzy_sets.shape[0], dtype=object)
        cdef int i

        for i in range(fuzzy_sets_copy.shape[0]):
            fuzzy_sets_copy[i] = copy.deepcopy(self.__fuzzy_sets[i])

        cdef LinguisticVariable new_object = LinguisticVariable(fuzzy_sets_copy, self.__concept, support_values_copy)
        memo[id(self)] = new_object
        return new_object