import numpy as np
cimport numpy as cnp

class LinguisticVariable:
    __support_values = None
    __fuzzy_sets = None
    __concept = None
    __domain = None

    def __init__(self, fuzzy_sets=None, name=None, support_values=None):
        if fuzzy_sets is None or len(fuzzy_sets) == 0:
            fuzzy_sets = []
            support_values = []
        elif support_values is None:
            raise TypeError('Support value cannot be None if fuzzy_sets is not None and not empty')

        self.__support_values = support_values
        self.__fuzzy_sets = fuzzy_sets
        self.__concept = name
        self.domain = [0, 1]

    def add_fuzzy_set(self, fuzzy_set, support_value):
        self.__fuzzy_sets.append(fuzzy_set)
        self.__support_values.append(support_value)

    def get_concept(self):
        return self.__concept

    def get_membership_value(self, fuzzy_set_index, x):
        if fuzzy_set_index > len(self.__fuzzy_sets):
            raise Exception(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")
        return self.__fuzzy_sets[fuzzy_set_index].get_membership_value(x)

    def get_length(self):
        return len(self.__fuzzy_sets)

    def get_fuzzy_set(self, fuzzy_set_index):
        return self.__fuzzy_sets[fuzzy_set_index]

    def get_support(self, fuzzy_set_id):
        return self.__support_values[fuzzy_set_id]
