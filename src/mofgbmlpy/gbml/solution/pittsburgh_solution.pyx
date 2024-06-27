import time

import numpy as np
cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classifier cimport Classifier
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution_builder cimport MichiganSolutionBuilder

cdef class PittsburghSolution(AbstractSolution):
    def __init__(self, num_vars, num_objectives, num_constraints, michigan_solution_builder, classifier):
        super().__init__(num_vars, num_objectives, num_constraints)
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classifier = classifier

        self._vars = michigan_solution_builder.create(num_vars)

    cdef MichiganSolutionBuilder get_michigan_solution_builder(self):
        return self.__michigan_solution_builder

    cpdef void remove_var(self, int index):
        self._vars = np.delete(self._vars, index)

    cpdef void clear_vars(self):
        self._vars = []

    cpdef void clear_attributes(self):
        self._attributes = {}

    cdef void learning(self):
        for var in self._vars:
            var.learning()

    cdef AbstractSolution classify(self, Pattern pattern):
        return self.__classifier.classify(self.get_vars(), pattern)

    cpdef double get_error_rate(self, dataset):
        error_rate, self.__errored_patterns = self.__classifier.get_error_rate(self.get_vars(), dataset)
        return error_rate

    cdef cnp.ndarray[object, ndim=1] get_errored_patterns(self):
        return self.__errored_patterns

    cpdef double compute_coverage(self):
        coverage = 0
        for michigan_solution in self._vars:
            coverage += michigan_solution.compute_coverage()
        return coverage

    cpdef int get_total_rule_weight(self): # TODO: get_total_rule_length or get_total_rule_weight ?
        return self.__classifier.get_rule_length(self._vars)
