# distutils: language = c++

import copy
cimport numpy as cnp
from libcpp cimport queue as cqueue
from libcpp cimport vector as cvector
from libc cimport math as cmath
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
import numpy as np


cdef class AllCombinationAntecedentFactory(AbstractAntecedentFactory):
    def __init__(self, knowledge):
        self.__dimension = knowledge.get_num_dim()
        self.generate_antecedents_indices(knowledge.get_fuzzy_vars())
        self.__knowledge = knowledge

    cdef void generate_antecedents_indices(self, FuzzyVariable[:] fuzzy_sets):
        cdef int i
        cdef int j
        cdef int k = 0
        cdef int current_dim
        cdef cqueue.queue[cvector.vector[int]] indices_queue
        cdef int num_generated_indices = 1
        cdef cvector.vector[int] tmp
        cdef FuzzyVariable var

        for i in range(self.__dimension):
            var = fuzzy_sets[i]
            num_generated_indices *= var.get_length()

        cdef int[:,:] indices = (np.empty((num_generated_indices, self.__dimension), dtype=int))

        indices_queue.push(cvector.vector[int]())

        # Generate all combination of fuzzy sets indices
        while indices_queue.size() > 0:
            buffer = indices_queue.front()
            indices_queue.pop()
            current_dim = buffer.size()
            if current_dim < self.__dimension:
                var = fuzzy_sets[current_dim]
                for i in range(var.get_length()):
                    tmp = cvector.vector[int]()

                    for j in range(current_dim):
                        tmp.push_back(buffer[j])
                    tmp.push_back(i)
                    indices_queue.push(tmp)
            else:
                # print(indices.shape[0], k)
                for i in range(self.__dimension):
                    indices[k][i] = buffer[i]
                k += 1

        self.__antecedents_indices = indices

    cdef Antecedent[:] create(self, int num_rules=1):
        cdef Antecedent[:] antecedent_objects = np.zeros(num_rules, dtype=object)
        cdef int[:,:] indices = self.create_antecedent_indices(num_rules)
        cdef int i
        cdef Antecedent new_antecedent_obj

        for i in range(num_rules):
            new_antecedent_obj = Antecedent(indices[i], self.__knowledge)
            antecedent_objects[i] = new_antecedent_obj

        return antecedent_objects

    cdef int[:,:] create_antecedent_indices(self, int num_rules=1):
        cdef int i
        cdef int j
        cdef int[:] chosen_list

        num_rules = min(num_rules, self.__antecedents_indices.shape[0])

        # Return an antecedent
        if self.__antecedents_indices is None:
            # with cython.gil:
            raise Exception("AllCombinationAntecedentFactory hasn't been initialised")
        cdef int[:] chosen_indices_lists = np.random.choice(np.arange(self.__antecedents_indices.shape[0], dtype=int), num_rules, replace=False)
        cdef int[:,:] new_indices = np.empty((num_rules, self.__dimension), dtype=int)

        for i in range(chosen_indices_lists.shape[0]):
            chosen_list = self.__antecedents_indices[chosen_indices_lists[i]]
            for j in range(chosen_list.shape[0]):
                new_indices[i][j] = chosen_list[j]
        return new_indices

    def __str__(self):
        return "AllCombinationAntecedentFactory [antecedents=" + str(self.__antecedents_indices) + ", dimension=" + str(
            self.__dimension) + "]"

    def __deepcopy__(self, memo={}):
        new_object = AllCombinationAntecedentFactory(self.__knowledge)

        memo[id(self)] = new_object
        return new_object
