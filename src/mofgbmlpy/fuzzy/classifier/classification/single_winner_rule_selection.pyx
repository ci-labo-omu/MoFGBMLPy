import time

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
cimport numpy as cnp

cdef class SingleWinnerRuleSelection(AbstractClassification):
    cpdef MichiganSolution classify(self, cnp.ndarray[object, ndim=1] michigan_solution_list, Pattern pattern):
        if len(michigan_solution_list) < 1:
            raise Exception("argument [michigan_solution_list] must contain at list 1 item")

        can_classify = False
        max = float('-inf')
        winner = michigan_solution_list[0]

        for solution in michigan_solution_list:
            if solution.get_class_label().is_rejected():
                raise Exception("one item in the argument [michigan_solution_list] has a rejected class label (it should not be used for classification)")
            value = solution.get_fitness_value(pattern.get_attributes_vector())

            if value > max:
                max = value
                winner = solution
                can_classify = True
            elif value == max:
                # There are 2 best solutions with the same fitness value
                if not solution.get_class_label() == winner.get_class_label():
                    can_classify = False

        if can_classify and max >= 0:
            return winner
        else:
            return None

    def __copy__(self):
        return SingleWinnerRuleSelection()

    def __str__(self):
        return self.__class__.__name__
