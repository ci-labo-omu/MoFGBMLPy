# distutils: language = c++

from libcpp.unordered_map cimport unordered_map as cmap
from libcpp.vector cimport vector as cvector
from libcpp.list cimport list as clist
import time

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cimport numpy as cnp

cdef class SingleWinnerRuleSelection(AbstractClassification):
    cdef cvector[cmap[int, double]] __cache
    # cdef cvector[clist[int]] __cache_order
    cdef int __max_num_solutions_cached

    cdef __get_cache(self, MichiganSolution solution, Pattern pattern)
    cdef __set_cache(self, MichiganSolution solution, Pattern pattern, value)
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern)