# distutils: language = c++

# from cython.operator import dereference, postincrement
from libcpp.unordered_map cimport unordered_map as cmap
from libcpp.list cimport list as clist
from libcpp.pair cimport pair as cpair
import time
import cython
import numpy as np
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
cimport numpy as cnp
from libc.math cimport INFINITY
from collections import OrderedDict
from mofgbmlpy.utility.lru_cache cimport LRUCache


cdef class SingleWinnerRuleSelection(AbstractClassification):
    def __init__(self, int num_patterns, cache_size_per_pattern=32):
        cdef int i

        for i in range(num_patterns):
            self.__cache.push_back(LRUCache(cache_size_per_pattern))

    cdef double get_fitness_value(self, MichiganSolution solution, Pattern pattern):
        cdef int solution_hash = hash(solution)
        cdef int pattern_id = pattern.get_id()
        cdef double value

        if self.__cache[pattern_id].has(solution_hash):
            # print("###")
            return self.__cache[pattern_id].get(solution_hash)
        else:
            value = solution.get_fitness_value(pattern.get_attributes_vector())
            self.__cache[pattern_id].put(solution_hash, value)
            # print("___", self.__cache[pattern_id].get_size(), self.__cache[pattern_id].get_max_size())
            return value

    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern):
        cdef double max = -INFINITY
        cdef MichiganSolution winner
        cdef double value
        cdef MichiganSolution solution
        cdef bint can_classify = False

        if michigan_solution_list.shape[0] < 1:
            raise Exception("argument [michigan_solution_list] must contain at least 1 item")
        winner =  michigan_solution_list[0]

        for i in range(michigan_solution_list.shape[0]):
            solution = michigan_solution_list[i]
            if solution.get_class_label().is_rejected():
                raise Exception("one item in the argument [michigan_solution_list] has a rejected class label (it should not be used for classification)")


            # value = self.get_fitness_value(solution, pattern)
            value = solution.get_fitness_value(pattern.get_attributes_vector())

            if value > max:
                max = value
                winner = solution
                can_classify = True
            elif value == max:
                # There are 2 best solutions with the same fitness value
                if not solution.get_class_label() == winner.get_class_label():
                    can_classify = False

        if can_classify and max >= 0:
            return winner
        else:
            return None

    def __deepcopy__(self, memo={}):
        new_object = SingleWinnerRuleSelection(self.__cache.size())
        memo[id(self)] = new_object
        return new_object

    def __str__(self):
        return self.__class__.__name__
