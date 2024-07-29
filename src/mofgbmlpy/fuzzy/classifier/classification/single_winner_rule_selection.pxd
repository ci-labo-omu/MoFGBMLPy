# distutils: language = c++

from libcpp.unordered_map cimport unordered_map as cmap
from libcpp.vector cimport vector as cvector
from libcpp.list cimport list as clist
import time

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cimport numpy as cnp
from mofgbmlpy.utility.lru_cache cimport LRUCache

cdef class SingleWinnerRuleSelection(AbstractClassification):
    cdef cvector[LRUCache] __cache

    cdef double get_fitness_value(self, MichiganSolution solution, Pattern pattern)
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern)