from src.fuzzy.fuzzy_term.abstract_fuzzy_term import AbstractFuzzyTerm


class FuzzyTermTriangular(AbstractFuzzyTerm):
    __a = None
    __b = None
    __c = None

    def __init__(self, left_corner_x, upper_corner_x, right_corner_x):
        super().__init__()
        self.__a = left_corner_x
        self.__b = upper_corner_x
        self.__c = right_corner_x

    def get_membership_value(self, x):
        if x == self.__b:
            return 1.0
        elif x <= self.__a or x >= self.__c:
            return 0.0
        elif x < self.__b:
            return (x-self.__a) / (self.__b-self.__a)
        else:
            return (self.__c-x) / (self.__c-self.__b)


