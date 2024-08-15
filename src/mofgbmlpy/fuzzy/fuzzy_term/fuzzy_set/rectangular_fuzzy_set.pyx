from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.rectangular_mf import RectangularMF
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType


cdef class RectangularFuzzySet(FuzzySet):
    """Rectangular fuzzy set """
    def __init__(self, left, right, id, term):
        """Constructor

        Args:
            left (double): X coordinate of the leftmost side of the rectangle: membership is equals to 0 before this point and 1 after it
            right (double): X coordinate of the leftmost side of the rectangle: membership is equals to 0 after this point and 1 before it
            id (int): ID of the fuzzy set
            term (str): Name of the fuzzy set (e.g. small)
        """
        super().__init__(function=RectangularMF(left, right), id=id, division_type=DivisionType.EQUAL_DIVISION, term=term)
