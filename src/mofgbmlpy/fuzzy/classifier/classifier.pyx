from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification import AbstractClassification
from mofgbmlpy.gbml.solution.abstract_solution import AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
import cython


cimport numpy as cnp

cdef class Classifier:
    def __init__(self, classification):
        if classification is None or not isinstance(classification, AbstractClassification):
            # with cython.gil:
            raise Exception("Invalid classification method")
        self._classification = classification

    cdef AbstractSolution classify(self, cnp.ndarray[object, ndim=1] michigan_solution_list, Pattern pattern):
        return self._classification.classify(michigan_solution_list, pattern)

    def __copy__(self):
        return Classifier(self._classification)

    @staticmethod
    def get_rule_length(michigan_solution_list):
        length = 0
        for item in michigan_solution_list:
            length += item.get_rule_length()
        return length

    cdef object get_error_rate(self, cnp.ndarray[object, ndim=1] michigan_solution_list, dataset):
        num_errors = 0
        errored_patterns = []

        for sol in michigan_solution_list:
            sol.reset_num_wins()
            sol.reset_fitness()

        for pattern in dataset.get_patterns():
            winner_solution = self.classify(michigan_solution_list, pattern)

            if winner_solution is None:
                num_errors += 1
                errored_patterns.append(pattern)
                continue

            winner_solution.inc_num_wins()

            if pattern.get_target_class() != winner_solution.get_class_label():
                num_errors += 1
                errored_patterns.append(pattern)
            else:
                winner_solution.inc_fitness()
        return num_errors / dataset.get_size(), errored_patterns

    @staticmethod
    def get_rule_num(michigan_solution_list):
        return len(michigan_solution_list)
