from mofgbmlpy.data.pattern cimport Pattern
cimport numpy as cnp
import cython


cdef class Dataset:
    cdef int __size
    cdef int __num_dim  # number of attributes
    cdef int __num_classes
    cdef object __patterns

    cpdef Pattern get_pattern(self, int index)
    cpdef cnp.ndarray[object, ndim=1] get_patterns(self)
    cpdef int get_num_dim(self)
    cpdef int get_num_classes(self)
    cpdef int get_size(self)
