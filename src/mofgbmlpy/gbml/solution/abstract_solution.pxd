from abc import ABC, abstractmethod
import numpy as np
cimport numpy as cnp
import cython


cdef class AbstractSolution:
    cdef float[:] _objectives
    # cdef float[:] __constraints
    cdef object _attributes

    cpdef float[:] get_objectives(self)
    # cpdef float[:] get_constraints(self)
    cpdef void set_attribute(self, str key, object value)
    cpdef object get_attribute(self, str key)
    cpdef bint has_attribute(self, str key)
    cpdef void set_objective(self, int index, float value)
    cpdef float get_objective(self, int index)
    cpdef int get_num_vars(self)
    cdef void clear_vars(self)
    # cpdef float get_constraint(self, int index)
    # cpdef void set_constraint(self, int index, float value)
    cpdef int get_num_objectives(self)
    cpdef int get_num_constraints(self)
    cpdef object get_attributes(self)
    cpdef void clear_attributes(self)
