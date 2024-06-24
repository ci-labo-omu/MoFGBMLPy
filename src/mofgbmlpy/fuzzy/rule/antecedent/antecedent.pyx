import numpy as np
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
cimport cython
cimport numpy as cnp

cdef class Antecedent:
    # cdef public object __antecedent_indices
    # cdef public Knowledge __knowledge

    def __init__(self, cnp.ndarray[int, ndim=1] antecedent_indices, Knowledge knowledge):
        self.__antecedent_indices = antecedent_indices
        self.__knowledge = knowledge

    cpdef int get_array_size(self):
        return self.__antecedent_indices.size

    cpdef cnp.ndarray[int, ndim=1] get_antecedent_indices(self):
        return self.__antecedent_indices

    cpdef void set_antecedent_indices(self, cnp.ndarray[int, ndim=1] new_indices):
        self.__antecedent_indices = new_indices

    cpdef cnp.ndarray[double, ndim=1] get_compatible_grade(self, cnp.ndarray[double, ndim=1] attribute_vector):
        cdef int i
        cdef int size = self.get_array_size()
        cdef cnp.ndarray[double, ndim=1] grade = np.zeros(size, dtype=np.float64)

        if size != attribute_vector.size:
            raise ValueError("antecedent_indices and attribute_vector must have the same length")

        for i in range(size):
            val = attribute_vector[i]
            if self.__antecedent_indices[i] < 0 and val < 0:
                # categorical
                grade[i] = 1.0 if self.__antecedent_indices[i] == round(val) else 0.0
            elif self.__antecedent_indices[i] > 0 and val >= 0:
                # numerical
                grade[i] = self.__knowledge.get_membership_value(val, i, self.__antecedent_indices[i])
            elif self.__antecedent_indices[i] == 0:
                # don't care
                grade[i] = 1.0
            else:
                raise ValueError("Illegal argument")

        return grade

    cdef double get_compatible_grade_value(self, cnp.ndarray[double, ndim=1] attribute_vector):
        cdef int i
        cdef int size = self.get_array_size()
        cdef double grade_value = 1
        cdef double val
        cdef int[:] antecedent_indices

        if size != attribute_vector.size:
            raise ValueError("antecedent_indices and attribute_vector must have the same length")

        for i in range(size):
            val = attribute_vector[i]
            if self.__antecedent_indices[i] < 0 and val < 0:
                # categorical
                grade_value *= 1.0 if self.__antecedent_indices[i] == round(val) else 0.0
            elif self.__antecedent_indices[i] > 0 and val >= 0:
                # numerical
                grade_value *= self.__knowledge.get_membership_value(val, i, self.__antecedent_indices[i])
            elif self.__antecedent_indices[i] == 0:
                continue
            else:
                raise ValueError("Illegal argument")

        return grade_value

    cpdef int get_length(self):
        return np.count_nonzero(self.__antecedent_indices)

    def __copy__(self):
        return Antecedent(self.__antecedent_indices, knowledge=self.__knowledge)