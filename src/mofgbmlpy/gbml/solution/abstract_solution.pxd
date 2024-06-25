from abc import ABC, abstractmethod
import numpy as np
cimport numpy as cnp
import cython


cdef class AbstractSolution:
    cdef object __objectives
    cdef object _vars
    cdef object __constraints
    cdef object _attributes

    cpdef cnp.ndarray[double, ndim=1] get_objectives(self)
    cpdef object get_vars(self)
    cpdef cnp.ndarray[double, ndim=1] get_constraints(self)
    cpdef void set_attribute(self, int id, object value)
    cpdef object get_attribute(self, int id)
    cpdef cnp.npy_bool has_attribute(self, int  id)
    cpdef void set_objective(self, int index, double value)
    cpdef double get_objective(self, int index)
    cpdef object get_var(self, int index)
    cpdef void set_var(self, int index, object value)
    cpdef void set_vars(self, object new_vars)
    cpdef cnp.ndarray[double, ndim=1] get_constraint(self, int index)
    cpdef void set_constraint(self, int index, double value)
    cpdef int get_num_vars(self)
    cpdef int get_num_objectives(self)
    cpdef int get_num_constraints(self)
    cpdef object get_attributes(self)
    cdef void clear_attributes(self)
    cdef void clear_vars(self)
    cpdef double compute_coverage(self)