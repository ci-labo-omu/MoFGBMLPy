cdef class FuzzySet:
    def __init__(self, function, term=""):
        self.__function = function
        self.__term = term

    def __str__(self):
        return f"Fuzzy set {self.__term}"

    cpdef double get_membership_value(self, double x):
        return self.__function.get_value(x)
