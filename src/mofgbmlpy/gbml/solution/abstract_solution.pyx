from abc import ABC, abstractmethod
import numpy as np
cimport numpy as cnp


cdef class AbstractSolution:
    def __init__(self, num_objectives, num_constraints=0):
        self._attributes = {}
        self._objectives = np.empty(num_objectives, dtype=np.float64)
        # self.__constraints = np.empty(num_constraints, dtype=np.float64)

    cpdef double[:] get_objectives(self):
        return self._objectives

    # cpdef double[:] get_constraints(self):
    #     return self.__constraints

    cpdef void set_attribute(self, str key, object value):
        self._attributes[key] = value

    cpdef object get_attribute(self, str key):
        return self._attributes[key]

    cpdef cnp.npy_bool has_attribute(self, str key):
        return key in self._attributes

    cpdef void set_objective(self, int index, double value):
        self._objectives[index] = value

    cpdef double get_objective(self, int index):
        return self._objectives[index]

    cpdef int get_num_vars(self):
        raise Exception("This class is abstract")

    cdef void clear_vars(self):
        raise Exception("This class is abstract")

    # cpdef double get_constraint(self, int index):
    #     return self.__constraints[index]
    #
    # cpdef void set_constraint(self, int index, double value):
    #     self.__constraints[index] = value

    cpdef int get_num_objectives(self):
        return self._objectives.shape[0]

    cpdef int get_num_constraints(self):
        # return self.__constraints.shape[0]
        # TODO
        return 0

    def __repr__(self):
        raise Exception("This class is abstract")

    def __eq__(self, other):
        if not isinstance(other, AbstractSolution):
            return False
        return np.array_equal(self.get_vars(), other.get_vars())

    def __hash__(self):
        raise Exception("This class is abstract")

    cpdef object get_attributes(self):
        return self._attributes

    cdef void clear_attributes(self):
        self._attributes = {}

    cpdef double compute_coverage(self):
        # with cython.gil:
        raise Exception("AbstractSolution is abstract")
