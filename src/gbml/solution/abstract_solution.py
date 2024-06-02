from abc import ABC
import numpy as np


class AbstractSolution(ABC):
    __objectives = None
    _vars = None
    __constraints = None
    _attributes = None

    def __init__(self, num_vars, num_objectives, num_constraints=0):
        self._attributes = {}
        self._vars = np.zeros(num_vars, dtype=object)
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