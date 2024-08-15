# distutils: language = c++

# from cython.operator import dereference, postincrement
from libcpp.unordered_map cimport unordered_map as cmap
from libcpp.list cimport list as clist
from libcpp.pair cimport pair as cpair
import time
import cython
import numpy as np
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
cimport numpy as cnp
from libc.math cimport INFINITY
# from mofgbmlpy.utility.lru_cache cimport LRUCache


cdef class SingleWinnerRuleSelection(AbstractClassification):
    """Class containing a method to compute a winner rule for a given pattern using the single winner rule selection """

    # def __init__(self, cache_size=0): # TODO: Check cache performance
        # if cache_size <= 0:
        #     self.__cache_size = 0
        # else:
        #     self.__cache = LRUCache(cache_size)
        #     self.__cache_size = cache_size

    # cdef double get_fitness_value(self, MichiganSolution solution, Pattern pattern):
    #     cdef int solution_hash = hash(solution)
    #     cdef int pattern_id = pattern.get_id()
    #     cdef double value
    #
    #     # print(pattern_id, solution_hash, self.__cache.combine_keys(pattern_id, solution_hash))
    #
    #     if self.__cache.has(pattern_id, solution_hash):
    #         # print("###")
    #         return self.__cache.get(pattern_id, solution_hash)
    #         # cached_value = self.__cache.get(pattern_id, solution_hash)
    #         # computed_value = solution.get_fitness_value(pattern.get_attributes_vector())
    #         # if cached_value != computed_value:
    #         #     raise Exception(f"DIFFERENT VALUE for pattern {pattern_id} sol {solution_hash} : {cached_value} != {computed_value}")
    #     else:
    #         value = solution.get_fitness_value(pattern.get_attributes_vector())
    #         self.__cache.put(pattern_id, solution_hash, value)
    #         # print("___", self.__cache.get_size(), self.__cache.get_max_size())
    #         return value

    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern):
        """Outputs the winner rule (the michigan solution) based on a single winner rule. 
        The winner is the solution with the maximum fitness for the given pattern.
        If there are 2 winner rules (same fitness values) with different conclusion classes then None or if there is 
        no rule with a positive or null fitness value then None is returned.
        If there are 2 winner rules (same fitness values) with the same conclusion class, then the first rule is returned
        
        Args:
            michigan_solution_list (MichiganSolution[], Pattern): List of rules from where a winner will be determined
            pattern (Pattern): Pattern used to compute the rules' fitness

        Returns:
            Winner rule or None if it can't be determined (negative fitness or 2 winners with different conclusion classes)
        """
        cdef double max = -INFINITY
        cdef MichiganSolution winner
        cdef double value
        cdef MichiganSolution solution
        cdef bint can_classify = False

        if michigan_solution_list.shape[0] < 1:
            raise Exception("argument [michigan_solution_list] must contain at least 1 item")
        winner =  michigan_solution_list[0]

        if pattern is None:
            raise Exception("Pattern is None")

        for i in range(michigan_solution_list.shape[0]):
            solution = michigan_solution_list[i]

            if solution is None:
                raise Exception("A solution is None in the list")

            if solution.get_class_label().is_rejected():
                raise Exception("one item in the argument [michigan_solution_list] has a rejected class label (it should not be used for classification)")

            # if self.__cache_size == 0:  # No cache
            value = solution.get_fitness_value(pattern.get_attributes_vector())
            # else:
            #     value = self.get_fitness_value(solution, pattern)

            if value > max:
                max = value
                winner = solution
                can_classify = True
            elif value == max:
                # There are 2 best solutions with the same fitness value
                if solution.get_class_label() != winner.get_class_label():
                    can_classify = False

        if can_classify and max >= 0:
            return winner
        else:
            return None

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        # new_object = SingleWinnerRuleSelection(self.__cache_size)
        new_object = SingleWinnerRuleSelection()
        memo[id(self)] = new_object
        return new_object

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return self.__class__.__name__

    def __eq__(self, other):
        """Check if another object is equal to this one

        Args:
            other (object): Object compared to this one

        Returns:
            bool: True if they are equal and False otherwise
        """
        return isinstance(other, SingleWinnerRuleSelection)