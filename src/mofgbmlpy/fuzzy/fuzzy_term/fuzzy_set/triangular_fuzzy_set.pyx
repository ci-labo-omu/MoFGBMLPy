from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType


cdef class TriangularFuzzySet(FuzzySet):
    def __init__(self, left, center, right, id, term):
        """Constructor

        Args:
            left (double): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 before it
            center (double): X coordinate of the vertex in the center of the triangle: membership is equals to 1 at this point
            right (double): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 after it
            id (int): ID of the fuzzy set
            term (str): Name of the fuzzy set (e.g. small)
        """
        super().__init__(function=TriangularMF(left, center, right), id=id, division_type=DivisionType.EQUAL_DIVISION, term=term)
