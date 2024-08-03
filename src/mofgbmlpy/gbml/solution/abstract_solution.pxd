from abc import ABC, abstractmethod
import numpy as np
cimport numpy as cnp
import cython


cdef class AbstractSolution:
    cdef double[:] _objectives
    # cdef double[:] __constraints
    cdef object _attributes

    cpdef double[:] get_objectives(self)
    # cpdef double[:] get_constraints(self)
    cpdef void set_attribute(self, str key, object value)
    cpdef object get_attribute(self, str key)
    cpdef bint has_attribute(self, str key)
    cpdef void set_objective(self, int index, double value)
    cpdef double get_objective(self, int index)
    cpdef int get_num_vars(self)
    cdef void clear_vars(self)
    # cpdef double get_constraint(self, int index)
    # cpdef void set_constraint(self, int index, double value)
    cpdef int get_num_objectives(self)
    cpdef int get_num_constraints(self)
    cpdef object get_attributes(self)
    cpdef void clear_attributes(self)
    cpdef double compute_coverage(self)