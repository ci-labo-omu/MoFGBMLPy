from libcpp.string cimport string as std_string

cdef extern from "core/data/class_label/abstract_class_label.hpp":
    cdef cppclass AbstractClassLabelCpp "AbstractClassLabel":
        AbstractClassLabelCpp() except +

        bint is_rejected() const;
        void set_rejected();
        std_string to_string() const

cdef class AbstractClassLabel:
    cdef AbstractClassLabelCpp * ptr

    cpdef bint is_rejected(self)
    cpdef void set_rejected(self)
    cdef AbstractClassLabelCpp * get_ptr(self)
