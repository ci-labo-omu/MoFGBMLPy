import numpy as np

from src.gbml.solution.abstract_solution import AbstractSolution
from src.fuzzy.knowledge.knowledge import Knowledge
from src.gbml.solution.abstract_solution import AbstractSolution


class PittsburghSolution(AbstractSolution):
    _bounds = None
    _rule = None
    _rule_builder = None
    _classifier = None
    michigan_solution_builder = None

    def __init__(self, num_vars, num_objectives, num_constraints, michigan_solution_builder, classifier):
        super().__init__(num_vars, num_objectives, num_constraints)
        self.michigan_solution_builder = michigan_solution_builder
        self._classifier = classifier

        self._vars = michigan_solution_builder.create_michigan_solution(num_vars)

    def get_michigan_solution_builder(self):
        return self.michigan_solution_builder

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
        return self._classifier.classify(self.get_vars(), pattern)