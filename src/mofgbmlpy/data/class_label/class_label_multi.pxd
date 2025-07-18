from libcpp.vector cimport vector
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel, AbstractClassLabelCpp

cdef extern from "data/class_label/class_label_multi.hpp":
    cdef cppclass ClassLabelMultiCpp "ClassLabelMulti"(AbstractClassLabelCpp):
        ClassLabelMultiCpp(const vector[int]& class_label) except +
        ClassLabelMultiCpp(const ClassLabelMultiCpp& other) except +

        int get_length() const
        void set_class_label_value(const vector[int]& class_label)
        const vector[int]& get_class_label_value() const
        int get_class_label_value_at(int index) const
        void set_class_label_vector(const vector[int]& class_label)

        bint operator ==(const AbstractClassLabelCpp& other) const
        ClassLabelMultiCpp * clone() const

cdef class ClassLabelMulti(AbstractClassLabel):
    cpdef int get_length(self)
    cpdef object get_class_label_value(self)
    cpdef int get_class_label_value_at(self, int index)
    cpdef void set_class_label_value(self, object class_label)
    cdef ClassLabelMultiCpp * get_multi_ptr(self)
    @staticmethod
    cdef ClassLabelMulti wrap(ClassLabelMultiCpp * ptr)