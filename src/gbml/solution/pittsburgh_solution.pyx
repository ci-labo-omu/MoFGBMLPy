import time

import numpy as np

from gbml.solution.abstract_solution import AbstractSolution
from fuzzy.knowledge.knowledge import Knowledge
from gbml.solution.abstract_solution import AbstractSolution


class PittsburghSolution(AbstractSolution):
    __classifier = None
    __michigan_solution_builder = None
    __errored_patterns = None

    def __init__(self, num_vars, num_objectives, num_constraints, michigan_solution_builder, classifier):
        super().__init__(num_vars, num_objectives, num_constraints)
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classifier = classifier

        self._vars = michigan_solution_builder.create(num_vars)

    def get_michigan_solution_builder(self):
        return self.__michigan_solution_builder

    def remove_var(self, index):
        del self._vars[index]

    def clear_vars(self):
        self._vars = []

    def clear_attributes(self):
        self._attributes = {}

    def learning(self):
        for var in self._vars:
            var.learning()

    def classify(self, pattern):
        return self.__classifier.classify(self.get_vars(), pattern)

    def get_error_rate(self, dataset):
        error_rate, self.__errored_patterns = self.__classifier.get_error_rate(self.get_vars(), dataset)
        return error_rate

    def get_errored_patterns(self):
        return self.__errored_patterns

    def compute_coverage(self):
        coverage = 0
        for michigan_solution in self._vars:
            coverage += michigan_solution.compute_coverage()
        return coverage

    def get_total_rule_weight(self):
        return self.__classifier.get_rule_length(self._vars)
