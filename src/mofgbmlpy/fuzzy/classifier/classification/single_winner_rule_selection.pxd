import time

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
cimport numpy as cnp

cdef class SingleWinnerRuleSelection(AbstractClassification):
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern)