from src.data.pattern import Pattern


class Dataset:
    __size = 0
    __n_dim = 0
    __c_num = 0
    __patterns = None

    def __init__(self, size, n_dim, c_num):
        if size <= 0 or n_dim <= 0 or c_num <= 0:
            raise ValueError("Incorrect input data set information")
        self.__size = size
        self.__n_dim = n_dim
        self.__c_num = c_num
        self.__patterns = []

    def add_pattern(self, pattern):
        self.__patterns.append(pattern)

    def get_pattern_at(self, index):
        return self.__patterns[index]

    def get_patterns(self):
        return self.__patterns

    def __str__(self):
        if len(self.__patterns) == 0:
            return "null"
        txt = f"{self.__size}, {self.__n_dim}, {self.__c_num}\n"
        for pattern in self.__patterns:
            txt += f"{pattern}\n"
        return txt

    def get_n_dim(self):
        return self.__n_dim

    def get_c_num(self):
        return self.__c_num

    def get_size(self):
        return self.__size
