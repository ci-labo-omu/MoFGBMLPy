import time

import cython
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classification.abstract_classification cimport AbstractClassification
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
cimport numpy as cnp
from libc.math cimport INFINITY

cdef class SingleWinnerRuleSelection(AbstractClassification):
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern):
        cdef double max = -INFINITY
        cdef MichiganSolution winner
        cdef double value
        cdef MichiganSolution solution
        cdef bint can_classify = False

        if michigan_solution_list.shape[0] < 1:
            raise Exception("argument [michigan_solution_list] must contain at least 1 item")
        winner =  michigan_solution_list[0]

        for solution in michigan_solution_list:
            if solution.get_class_label().is_rejected():
                # with cython.gil:
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

    def __deepcopy__(self, memo={}):
        new_object = SingleWinnerRuleSelection()
        memo[id(self)] = new_object
        return new_object

    def __str__(self):
        return self.__class__.__name__
