from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
cimport numpy as cnp
import cython


cdef class ClassLabelMulti(AbstractClassLabel):
    def __init__(self, class_label):
        self.__class_label = class_label
        super().__init__()

    def __eq__(self, other):
        if not isinstance(other, ClassLabelMulti) or self.get_length() != other.get_length():
            return False

        cdef cnp.ndarray[int, ndim=1] label = self.get_class_label_value()
        cdef cnp.ndarray[int, ndim=1] other_label = other.get_class_label_value()
        cdef int i

        for i in range(self.get_length()):
            if label[i] != other_label[i]:
                return False
        return True

    cpdef int get_length(self):
        return self.get_class_label_value().size

    def __copy__(self):
        return ClassLabelMulti(self.get_class_label_value())

    def __str__(self):
        if self.get_class_label_value() is None:
            # with cython.gil:
            raise Exception("class label value is None")
        cdef cnp.ndarray[int, ndim=1] label_value = self.get_class_label_value()
        txt = f"{self.label_value[0]:2d}"

        if self.get_length() > 1:
            for i in range(1, self.get_length()):
                txt = f"{txt}, {label_value[i]:2d}"

        return txt

    cpdef object get_class_label_value(self):
        return self.__class_label

    cpdef void set_class_label_value(self, object class_label):
        self.__class_label = class_label

