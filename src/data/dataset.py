from data.pattern import Pattern
import numpy as np


class Dataset:
    __size = 0
    __num_dim = 0  # number of attributes
    __num_classes = 0
    __patterns = None

    def __init__(self, size, n_dim, c_num, patterns):
        if size <= 0 or n_dim <= 0 or c_num <= 0:
            raise ValueError("Incorrect input data set information")
        self.__size = size
        self.__num_dim = n_dim
        self.__num_classes = c_num
        self.__patterns = np.array(patterns)

    def get_pattern(self, index):
        return self.__patterns[index]

    def get_patterns(self):
        return self.__patterns

    def __str__(self):
        if len(self.__patterns) == 0:
            return "null"
        txt = f"{self.__size}, {self.__num_dim}, {self.__num_classes}\n"
        for pattern in self.__patterns:
            txt += f"{pattern}\n"
        return txt

    def get_num_dim(self):
        return self.__num_dim

    def get_num_classes(self):
        return self.__num_classes

    def get_size(self):
        return self.__size
