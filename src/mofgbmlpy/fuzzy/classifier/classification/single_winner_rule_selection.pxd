# distutils: language = c++

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

from mofgbmlpy.utility.lru_cache cimport LRUCache

cdef class SingleWinnerRuleSelection(AbstractClassification):
    cdef LRUCache __cache
    cdef int __cache_size

    cdef double get_fitness_value(self, MichiganSolution solution, Pattern pattern)
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern)