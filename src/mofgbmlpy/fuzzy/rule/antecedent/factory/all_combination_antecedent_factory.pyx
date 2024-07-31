# distutils: language = c++

import copy
cimport numpy as cnp
from libcpp.queue cimport queue as cqueue
from libcpp.vector cimport vector as cvector
from libc cimport math as cmath
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
import numpy as np


cdef class AllCombinationAntecedentFactory(AbstractAntecedentFactory):
    def __init__(self, knowledge):
        if knowledge is None or knowledge.get_num_dim() == 0:
            raise Exception("knowledge can't be None and must have at least one fuzzy variable")

        self.__knowledge = knowledge
        self.__antecedents_indices = self.generate_antecedents_indices()

    def get_num_antecedents(self):
        return len(self.__antecedents_indices)

    cdef int[:,:] generate_antecedents_indices(self):
        cdef int i
        cdef int j
        cdef int k = 0
        cdef int current_dim
        cdef cqueue[cvector[int]] indices_queue
        cdef int num_generated_indices = 1
        cdef cvector[int] tmp
        cdef FuzzyVariable var
        cdef int dimension = self.__knowledge.get_num_dim()
        cdef FuzzyVariable[:] fuzzy_vars = self.__knowledge.get_fuzzy_vars()
        
        for i in range(dimension):
            var = fuzzy_vars[i]
            num_generated_indices *= var.get_length()

        cdef int[:,:] indices = (np.empty((num_generated_indices, dimension), dtype=int))

        indices_queue.push(cvector[int]())

        # Generate all combination of fuzzy sets indices
        while indices_queue.size() > 0:
            buffer = indices_queue.front()
            indices_queue.pop()
            current_dim = buffer.size()
            if current_dim < dimension:
                var = fuzzy_vars[current_dim]
                for i in range(var.get_length()):
                    tmp = cvector[int]()

                    for j in range(current_dim):
                        tmp.push_back(buffer[j])
                    tmp.push_back(i)
                    indices_queue.push(tmp)
            else:
                # print(indices.shape[0], k)
                for i in range(dimension):
                    indices[k][i] = buffer[i]
                k += 1

        return indices

    def generate_antecedents_indices_py(self):
        self.generate_antecedents_indices()

    cdef Antecedent[:] create(self, int num_rules=1):
        cdef int[:,:] indices = self.create_antecedent_indices(num_rules)
        cdef Antecedent[:] antecedent_objects = np.zeros(num_rules, dtype=object)
        cdef int i
        cdef Antecedent new_antecedent_obj

        for i in range(num_rules):
            new_antecedent_obj = Antecedent(indices[i], self.__knowledge)
            antecedent_objects[i] = new_antecedent_obj

        return antecedent_objects

    def create_py(self, int num_rules=1):
        return self.create(num_rules)


    cdef int[:,:] create_antecedent_indices(self, int num_rules=1):
        cdef int i
        cdef int j
        cdef int[:] chosen_list

        if num_rules <= 0:
            raise Exception("num_rules must be positive")

        num_rules = min(num_rules, self.__antecedents_indices.shape[0])

        # Return an antecedent
        cdef int[:] chosen_indices_lists = np.random.choice(np.arange(self.__antecedents_indices.shape[0], dtype=int), num_rules, replace=False)
        cdef int[:,:] new_indices = np.empty((num_rules, self.__knowledge.get_num_dim()), dtype=int)

        for i in range(chosen_indices_lists.shape[0]):
            chosen_list = self.__antecedents_indices[chosen_indices_lists[i]]
            for j in range(chosen_list.shape[0]):
                new_indices[i][j] = chosen_list[j]
        return new_indices

    def create_antecedent_indices_py(self, int num_rules=1):
        return self.create_antecedent_indices(num_rules)

    def __str__(self):
        return "AllCombinationAntecedentFactory [antecedents=" + str(self.__antecedents_indices) + ", dimension=" + str(
            self.__knowledge.get_num_dim()) + "]"

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, AllCombinationAntecedentFactory):
            return False

        return self.__knowledge == other.get_knowledge()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            Deep copy of this object
        """
        new_object = AllCombinationAntecedentFactory(self.__knowledge)

        memo[id(self)] = new_object
        return new_object

    cpdef Knowledge get_knowledge(self):
        return self.__knowledge