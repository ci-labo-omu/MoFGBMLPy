# distutils: language = c++
import copy

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
import cython
import numpy as np
from libcpp.vector cimport vector as cvector
cimport numpy as cnp

cdef class Classifier:
    def __init__(self, classification):
        if classification is None or not isinstance(classification, AbstractClassification):
            # with cython.gil:
            raise Exception("Invalid classification method")
        self._classification = classification

    cdef AbstractSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern):
        return self._classification.classify(michigan_solution_list, pattern)

    def __copy__(self):
        return Classifier(self._classification)

    def __deepcopy__(self, memo={}):
        new_object = Classifier(copy.deepcopy(self._classification))
        memo[id(self)] = new_object
        return new_object

    @staticmethod
    def get_length(michigan_solution_list):
        length = 0
        for item in michigan_solution_list:
            length += item.get_length()
        return length

    cdef double get_error_rate(self, MichiganSolution[:] michigan_solution_list, Dataset dataset):
        cdef int num_errors = 0
        cdef int dataset_size = dataset.get_size()
        cdef int i
        cdef AbstractSolution winner_solution
        cdef Pattern[:] patterns = dataset.get_patterns()
        cdef Pattern p

        for sol in michigan_solution_list:
            sol.reset_num_wins()
            sol.reset_fitness()

        for i in range(dataset.get_size()):
            p = patterns[i]
            winner_solution = self.classify(michigan_solution_list, p)

            if winner_solution is None:
                num_errors += 1
                continue

            winner_solution.inc_num_wins()

            if p.get_target_class() != winner_solution.get_class_label():
                num_errors += 1
            else:
                winner_solution.inc_fitness()

        return num_errors / dataset_size

    cdef object[:] get_errored_patterns(self, MichiganSolution[:] michigan_solution_list, Dataset dataset):
        cdef int i
        cdef cvector[int] errored_patterns_indices
        cdef object[:] errored_patterns
        cdef AbstractSolution winner_solution
        cdef Pattern[:] patterns = dataset.get_patterns()
        cdef Pattern p

        for i in range(dataset.get_size()):
            p = patterns[i]
            winner_solution = self.classify(michigan_solution_list, p)

            if winner_solution is None or p.get_target_class() != winner_solution.get_class_label():
                errored_patterns_indices.push_back(i)

        errored_patterns = np.empty(errored_patterns_indices.size(), dtype=object)
        for i in range(errored_patterns_indices.size()):
            errored_patterns[i] = patterns[errored_patterns_indices[i]]

        return errored_patterns

    @staticmethod
    def get_rule_num(michigan_solution_list):
        return len(michigan_solution_list)
