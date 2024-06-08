import numpy as np

from src.gbml.solution.abstract_solution import AbstractSolution
from src.fuzzy.knowledge.knowledge import Knowledge
from src.gbml.solution.abstract_solution import AbstractSolution


class PittsburghSolution(AbstractSolution):
    __classifier = None
    __michigan_solution_builder = None

    def __init__(self, num_vars, num_objectives, num_constraints, michigan_solution_builder, classifier):
        super().__init__(num_vars, num_objectives, num_constraints)
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classifier = classifier

        self._vars = michigan_solution_builder.create_michigan_solution(num_vars)

    def get_michigan_solution_builder(self):
        return self.__michigan_solution_builder

    def remove_var(self, index):
        del self._vars[index]

    def add_var(self, value):
        self._vars.append(value)

    def clear_vars(self):
        self._vars = []

    def clear_attributes(self):
        self._attributes = {}

    def learning(self):
        for var in self._vars:
            var.learning()

    def classify(self, pattern):
        return self.__classifier.classify(self.get_vars(), pattern)

    def get_error_rate(self, training_set):
        num_errors = 0
        for pattern in training_set.get_patterns():
            winner_solution = self.classify(pattern)

            if winner_solution is None:
                num_errors += 1
                continue

            if pattern.get_target_class() != winner_solution.get_class_label():
                num_errors += 1

        return num_errors / training_set.get_size()
