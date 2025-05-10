import numpy as np
from mofgbmlpy.data.pattern cimport Pattern
cimport numpy as cnp
import cython
from mofgbmlpy.data.dataset_density cimport DatasetWithDensity

cdef class DatasetManager:
    cdef list datasets  # Cython用の属性（Cレベルで直接アクセス）
    cdef int current_index
    cdef public DatasetWithDensity current_dataset
    cdef object algorithm
    cpdef void switch_dataset(self)
    cpdef DatasetWithDensity get_current_dataset(self)
    cpdef set_algorithm(self, algorithm)

