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


cdef class SingleWinnerRuleSelection(AbstractClassification):
    def __init__(self, int num_patterns):
        self.__max_num_solutions_cached = 30
        cdef int i

        for i in range(num_patterns):
            self.__cache.push_back(cmap[int, double]())
            # self.__cache_order.push_back(clist[int]())

    cdef __get_cache(self, MichiganSolution solution, Pattern pattern):
        cdef int solution_hash = hash(solution)
        cdef int pattern_id = pattern.get_id()
        cdef cmap[int, double] pattern_cache = self.__cache[pattern_id]
        # cdef clist[int] pattern_cache_order = self.__cache_order[pattern_id]
        cdef double value
        cdef int key

        if pattern_cache.find(solution_hash) != pattern_cache.end():
            value = pattern_cache[solution_hash]

            # Put it back at the back of the "queue" because we don't want it to be deleted soon since it's maybe often used
            # list_start = pattern_cache_order.begin()
            #
            # while dereference(list_start) != solution_hash and list_start != self.__cache_order[pattern_id].end():
            #     postincrement(list_start)
            #
            # if list_start != self.__cache_order[pattern_id].end():
            #     key = dereference(list_start)
            #     pattern_cache_order.erase(list_start)
            #     pattern_cache_order.push_back(key)
            # else:
            #     raise Exception("Cache data structures (orders and values) are not synced")
            # print("###")
            return value
        # print("___")

        return None

    cdef __set_cache(self, MichiganSolution solution, Pattern pattern, value):
        cdef long int solution_hash = hash(solution)
        cdef int pattern_id = pattern.get_id()

        if self.__cache[pattern_id].size() == self.__max_num_solutions_cached:
            # Cache is full so remove the cached item at the front

            # 1. Update cache order
            # key = self.__cache_order[pattern_id].front()
            # self.__cache_order[pattern_id].pop_front()
            # self.__cache_order[pattern_id].push_back(solution_hash)

            # 2. Update cache content
            self.__cache[pattern_id].erase(self.__cache[pattern_id].begin())
            self.__cache[pattern_id].insert(cpair.pair[int, double](solution_hash, value))
        else:
            self.__cache[pattern_id][solution_hash] = value
            # self.__cache_order[pattern_id].push_back(solution_hash)

    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern):
        cdef double max = -INFINITY
        cdef MichiganSolution winner
        cdef double value
        cdef MichiganSolution solution
        cdef bint can_classify = False

        if michigan_solution_list.shape[0] < 1:
            raise Exception("argument [michigan_solution_list] must contain at least 1 item")
        winner =  michigan_solution_list[0]

        for solution in michigan_solution_list:
            if solution.get_class_label().is_rejected():
                # with cython.gil:
                raise Exception("one item in the argument [michigan_solution_list] has a rejected class label (it should not be used for classification)")

            cached_value = self.__get_cache(solution, pattern)
            if cached_value is not None:
                value = cached_value
            else:
                value = solution.get_fitness_value(pattern.get_attributes_vector())
                self.__set_cache(solution, pattern, value)

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
        new_object = SingleWinnerRuleSelection(len(self.__cache))
        memo[id(self)] = new_object
        return new_object

    def __str__(self):
        return self.__class__.__name__
