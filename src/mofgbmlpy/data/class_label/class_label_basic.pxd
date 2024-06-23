from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel


cdef class ClassLabelBasic(AbstractClassLabel):
    cdef int __class_label
    cpdef object get_class_label_value(self)
    cpdef void set_class_label_value(self, object class_label)
