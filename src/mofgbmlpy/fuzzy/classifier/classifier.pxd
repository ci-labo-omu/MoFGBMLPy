from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification import AbstractClassification
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution

cimport numpy as cnp

cdef class Classifier:
    cdef object _classification

    cdef AbstractSolution classify(self, cnp.ndarray[object, ndim=1] michigan_solution_list, Pattern pattern)
    cdef object get_error_rate(self, cnp.ndarray[object, ndim=1] michigan_solution_list, dataset)