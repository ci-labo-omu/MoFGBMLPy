cdef class FuzzySet:
    def __init__(self, function, term=""):
        self.__function = function
        self.__term = term

    def __repr__(self):
        return f"Fuzzy set {self.__term}"

    cdef double get_membership_value(self, double x):
        return self.__function.get_value(x)
