import copy

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType


cdef class TriangularFuzzySet(FuzzySet):
    def __init__(self, left, center, right, id, term):
        """Constructor

        Args:
            left (float): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 before it
            center (float): X coordinate of the vertex in the center of the triangle: membership is equals to 1 at this point
            right (float): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 after it
            id (int): ID of the fuzzy set
            term (str): Name of the fuzzy set (e.g. small)
        """
        super().__init__(function=TriangularMF(left, center, right), id=id, division_type=DivisionType.EQUAL_DIVISION, term=term)

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef float[:] params = self.get_function().get_params()
        cdef TriangularFuzzySet new_object = TriangularFuzzySet(params[0], params[1], params[2], self._id, self._term)

        memo[id(self)] = new_object
        return new_object