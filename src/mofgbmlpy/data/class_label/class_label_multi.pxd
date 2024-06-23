from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
cimport numpy as cnp


cdef class ClassLabelMulti(AbstractClassLabel):
    cdef object __class_label

    cpdef int get_length(self)

    cpdef object get_class_label_value(self)
    cpdef void set_class_label_value(self, object class_label)
