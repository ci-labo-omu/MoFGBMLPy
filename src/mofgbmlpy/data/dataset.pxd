from mofgbmlpy.data.pattern cimport Pattern
cimport numpy as cnp
import cython
from libcpp.vector cimport vector
from libcpp.string cimport string as std_string
from mofgbmlpy.data.pattern cimport PatternCpp

cdef extern from "core/data/dataset.hpp":
    cdef cppclass DatasetCpp "Dataset":
        DatasetCpp(int size, int num_dim, int num_classes, const vector[PatternCpp*] &patterns) except +
        DatasetCpp(const DatasetCpp& other) except +

        PatternCpp * get_pattern(int index) const
        const vector[PatternCpp*]& get_patterns() const
        int get_num_dim() const
        int get_num_classes() const
        int get_size() const

        std_string to_string() const
        bint operator ==(const DatasetCpp& other) const
        DatasetCpp * clone() const

cdef class Dataset:
    cdef DatasetCpp* ptr

    cpdef Pattern get_pattern(self, int index)
    cpdef Pattern[:] get_patterns(self)
    cpdef int get_num_dim(self)
    cpdef int get_num_classes(self)
    cpdef int get_size(self)
    @staticmethod
    cdef Dataset wrap(DatasetCpp * wrapped_ptr)
