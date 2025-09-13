import copy

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.rectangular_mf import RectangularMF
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf import AbstractMF

cdef class RectangularFuzzySet(FuzzySet):
    """Rectangular fuzzy set """
    def __init__(self, left, right, id, term):
        """Constructor

        Args:
            left (float): X coordinate of the leftmost side of the rectangle: membership is equals to 0 before this point and 1 after it
            right (float): X coordinate of the leftmost side of the rectangle: membership is equals to 0 after this point and 1 before it
            id (int): ID of the fuzzy set
            term (str): Name of the fuzzy set (e.g. small)
        """
        super().__init__(function=RectangularMF(left, right), id=id, division_type=DivisionType.EQUAL_DIVISION, term=term)

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef float[:] params = self.get_function().get_params()
        cdef RectangularFuzzySet new_object = RectangularFuzzySet(params[0], params[1], self._id, self._term)

        memo[id(self)] = new_object
        return new_object