from abc import ABC, abstractmethod
import numpy as np
cimport numpy as cnp


cdef class AbstractSolution:
    def __init__(self, num_vars, num_objectives, num_constraints=0):
        self._attributes = {}
        self._vars = []
        self._objectives = np.empty(num_objectives, dtype=np.float64)
        self.__constraints = np.empty(num_constraints, dtype=np.float64)

    cpdef double[:] get_objectives(self):
        return self._objectives

    cpdef object get_vars(self):
        return self._vars

    cpdef double[:] get_constraints(self):
        return self.__constraints

    cpdef void set_attribute(self, int id, object value):
        self._attributes[id] = value

    cpdef object get_attribute(self, int id):
        return self._attributes[id]

    cpdef cnp.npy_bool has_attribute(self, int id):
        return id in self._attributes

    cpdef void set_objective(self, int index, double value):
        self._objectives[index] = value

    cpdef double get_objective(self, int index):
        return self._objectives[index]

    cpdef object get_var(self, int index):
        return self._vars[index]

    cpdef void set_var(self, int index, object value):
        self._vars[index] = value

    cpdef void set_vars(self, object new_vars):
        self._vars = new_vars

    cpdef cnp.ndarray[double, ndim=1] get_constraint(self, int index):
        return self.__constraints[index]

    cpdef void set_constraint(self, int index, double value):
        self.__constraints[index] = value

    cpdef int get_num_vars(self):
        return self._vars.size

    cpdef int get_num_objectives(self):
        return self._objectives.size

    cpdef int get_num_constraints(self):
        return self.__constraints.size

    def __repr__(self):
        txt = "Variables: ["
        for i in range(self.get_num_vars()):
            txt += f"{self._vars[i]} "

        txt += "] Objectives "
        for i in range(self.get_num_objectives()):
            txt += f"{self._objectives[i]} "

        txt += "] Constraints "
        for i in range(self.get_num_objectives()):
            txt += f"{self._objectives[i]} "

        txt += f"] Algorithm Attributes: {self._attributes}"

        return txt

    def __eq__(self, other):
        if not isinstance(other, AbstractSolution):
            return False
        return np.array_equal(self.get_vars(), other.get_vars())

    def __hash__(self):
        return hash(self._vars)

    cpdef object get_attributes(self):
        return self._attributes

    cdef void clear_attributes(self):
        self._attributes = {}

    cdef void clear_vars(self):
        self._vars = []

    cpdef double compute_coverage(self):
        # with cython.gil:
        raise Exception("AbstractSolution is abstract")
