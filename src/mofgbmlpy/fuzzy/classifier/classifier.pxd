from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
cimport numpy as cnp
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class Classifier:
    cdef AbstractClassification _classification

    cdef AbstractSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern)
    cpdef AbstractSolution classify_py(self, MichiganSolution[:] michigan_solution_list, Pattern pattern)
    cdef double get_error_rate(self, MichiganSolution[:] michigan_solution_list, Dataset dataset)
    cpdef double get_error_rate_py(self, MichiganSolution[:] michigan_solution_list, Dataset dataset)
    cdef object[:] get_errored_patterns(self, MichiganSolution[:] michigan_solution_list, Dataset dataset)
    cpdef object[:] get_errored_patterns_py(self, MichiganSolution[:] michigan_solution_list, Dataset dataset)
    cpdef AbstractClassification get_classification(self)