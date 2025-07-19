from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel, AbstractClassLabelCpp
cimport cython

cdef extern from "core/data/class_label/class_label_basic.hpp":
    cdef cppclass ClassLabelBasicCpp "ClassLabelBasic"(AbstractClassLabelCpp):
        ClassLabelBasicCpp(int class_label) except +
        ClassLabelBasicCpp(const ClassLabelBasicCpp& other)  except +

        int get_class_label_value() const
        void set_class_label_value(int class_label)


        bint operator ==(const AbstractClassLabelCpp& other) const
        ClassLabelBasicCpp * clone() const

cdef class ClassLabelBasic(AbstractClassLabel):
    cpdef int get_class_label_value(self)
    cpdef void set_class_label_value(self, int class_label)
    cdef ClassLabelBasicCpp * get_basic_ptr(self)
    @staticmethod
    cdef ClassLabelBasic wrap(ClassLabelBasicCpp * ptr)