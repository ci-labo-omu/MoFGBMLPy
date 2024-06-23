import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set cimport FuzzySet


cdef class LinguisticVariable:
    def __init__(self, fuzzy_sets=None, name=None, support_values=None):
        if fuzzy_sets is None or len(fuzzy_sets) == 0:
            fuzzy_sets = []
            support_values = []
        elif support_values is None:
            raise TypeError('Support value cannot be None if fuzzy_sets is not None and not empty')

        self.__support_values = support_values
        self.__fuzzy_sets = fuzzy_sets
        self.__concept = str(name)
        self.__domain = [0, 1]

    cpdef void add_fuzzy_set(self, FuzzySet fuzzy_set, double support_value):
        self.__fuzzy_sets.append(fuzzy_set)
        self.__support_values.append(support_value)

    cpdef str get_concept(self):
        return self.__concept

    cpdef double get_membership_value(self, int fuzzy_set_index, double x):
        if fuzzy_set_index > len(self.__fuzzy_sets):
            raise Exception(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")
        return self.__fuzzy_sets[fuzzy_set_index].get_membership_value(x)

    cpdef int get_length(self):
        return len(self.__fuzzy_sets)

    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index):
        return self.__fuzzy_sets[fuzzy_set_index]

    cpdef double get_support(self, int fuzzy_set_id):
        return self.__support_values[fuzzy_set_id]
