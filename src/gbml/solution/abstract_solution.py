from abc import ABC, abstractmethod
import numpy as np


class AbstractSolution(ABC):
    __objectives = None
    _vars = None
    __constraints = None
    _attributes = None

    def __init__(self, num_vars, num_objectives, num_constraints=0):
        self._attributes = {}
        self._vars = []
        self.__objectives = np.zeros(num_objectives, dtype=np.float_)
        self.__constraints = np.zeros(num_constraints, dtype=np.float_)

    def get_objectives(self):
        return self.__objectives

    def get_vars(self):
        return self._vars

    def get_constraints(self):
        return self.__constraints

    def set_attribute(self, id, value):
        self._attributes[id] = value

    def get_attribute(self, id):
        return self._attributes[id]

    def has_attribute(self, id):
        return id in self._attributes

    def set_objective(self, index, value):
        self.__objectives[index] = value

    def get_objective(self, index):
        return self.__objectives[index]

    def get_var(self, index):
        return self._vars[index]

    def set_var(self, index, value):
        self._vars[index] = value

    def get_constraint(self, index):
        return self.__constraints[index]

    def set_constraint(self, index, value):
        self.__constraints[index] = value

    def get_num_vars(self):
        return len(self._vars)

    def get_num_objectives(self):
        return len(self.__objectives)

    def get_num_constraints(self):
        return len(self.__constraints)

    def __str__(self):
        return f"Variables: {self._vars} Objectives {self.__objectives} Constraints {self.__constraints}\tAlgorithm Attributes: {self._attributes}"

    def __eq__(self, other):
        if not isinstance(other, AbstractSolution):
            return False
        return self.get_vars().__eq__(other.get_vars())

    def __hash__(self):
        return hash(self._vars)

    def get_attributes(self):
        return self._attributes

    def clear_attributes(self):
        self._attributes = {}

    def clear_vars(self):
        self._vars = []

    @abstractmethod
    def compute_coverage(self):
        pass

    class SolutionBuilderCore(ABC):
        _bounds = None
        _num_objectives = None
        _num_constraints = None
        _rule_builder = None

        def __init__(self, bounds, num_objectives, num_constraints, rule_builder):
            self._bounds = bounds
            self._num_objectives = num_objectives
            self._num_constraints = num_constraints
            self._rule_builder = rule_builder
