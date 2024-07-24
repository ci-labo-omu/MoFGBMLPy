# distutils: language = c++

from libcpp cimport map as cmap
from libcpp cimport vector as cvector
from libcpp cimport queue as cqueue
import time

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cimport numpy as cnp

cdef class SingleWinnerRuleSelection(AbstractClassification):
    cdef cvector.vector[cmap.map[int, double]] __cache
    cdef cvector.vector[cqueue.queue[int]] __cache_order
    cdef int[:] __cache_current_solution_index
    cdef int __max_num_solutions_cached

    cpdef __get_cache(self, MichiganSolution solution, Pattern pattern)
    cpdef __set_cache(self, MichiganSolution solution, Pattern pattern, value)
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern)