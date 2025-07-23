cimport numpy as cnp
import cython
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabelCpp
from libcpp.vector cimport vector
from libcpp.string cimport string as std_string


cdef extern from "core/data/pattern.hpp":
    cdef cppclass PatternCpp "Pattern":
        PatternCpp(int id, const vector[double] &attributes_vector, AbstractClassLabelCpp* target_class) except +
        PatternCpp(const PatternCpp& other) except +

        int get_id() const
        const vector[double]& get_attributes_vector() const
        double get_attribute_value(int index) except +
        AbstractClassLabelCpp* get_target_class() const
        int get_num_dim() const
        void set_attribute_value(int index, float new_value) except +

        bint operator==(const PatternCpp& other) const
        std_string to_string() const
        PatternCpp* clone() const

cdef class Pattern:
    cdef PatternCpp* ptr

    cpdef int get_id(self)
    cpdef double[:] get_attributes_vector(self)
    cpdef double get_attribute_value(self, int index)
    cpdef object get_target_class(self)
    cpdef int get_num_dim(self)
    cpdef void set_attribute_value(self, int index, float new_value)
    @staticmethod
    cdef Pattern wrap(PatternCpp * ptr)