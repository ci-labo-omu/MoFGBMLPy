from libcpp.vector cimport vector
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel, AbstractClassLabelCpp

cdef extern from "core/data/class_label/class_label_multi.hpp":
    cdef cppclass ClassLabelMultiCpp "ClassLabelMulti"(AbstractClassLabelCpp):
        ClassLabelMultiCpp(const vector[int]& class_label) except +
        ClassLabelMultiCpp(const ClassLabelMultiCpp& other) except +

        int get_length() const
        void set_class_label_value(const vector[int]& class_label)
        const vector[int]& get_class_label_value() const
        int get_class_label_value_at(int index) except +
        void set_class_label_vector(const vector[int]& class_label)
        void set_class_label_value_at(const int index, const int new_value) except +

cdef class ClassLabelMulti(AbstractClassLabel):
    cpdef int get_length(self)
    cpdef object get_class_label_value(self)
    cpdef int get_class_label_value_at(self, int index)
    cpdef void set_class_label_value(self, object class_label)
    cpdef void set_class_label_value_at(self, int index, int value)
    cdef ClassLabelMultiCpp * get_multi_ptr(self)
    @staticmethod
    cdef ClassLabelMulti wrap(ClassLabelMultiCpp * ptr)