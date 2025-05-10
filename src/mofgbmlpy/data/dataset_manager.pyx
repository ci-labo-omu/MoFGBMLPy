import numpy as np
cimport numpy as cnp
from mofgbmlpy.data.dataset_density cimport DatasetWithDensity
from numpy cimport ndarray
import cython
import gc

cdef class DatasetManager:

    def __init__(self, datasets):  # Pythonレベルでアクセス可能
        self.datasets = datasets
        self.current_dataset = datasets[0]
        self.current_index = 0

    cpdef void switch_dataset(self):  # Cython/Python両方でアクセス可能
        current_gen = self.algorithm.n_gen
        if 0 <= self.current_index < len(self.datasets)-1:
            self.current_index += 1
            print(f"Switching dataset at generation {current_gen}")
        else:
            pass


    cpdef DatasetWithDensity get_current_dataset(self):  # Cython/Python両方でアクセス可能
        gc.collect()
        return self.datasets[self.current_index]

    cpdef set_algorithm(self, algorithm):
        self.algorithm = algorithm