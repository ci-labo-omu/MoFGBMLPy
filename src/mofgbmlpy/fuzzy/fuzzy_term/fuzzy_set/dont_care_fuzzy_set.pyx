from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.dont_care_mf import DontCareMF
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType


cdef class DontCareFuzzySet(FuzzySet):
    def __init__(self, id):
        dont_care_mf = DontCareMF()
        super().__init__(function=dont_care_mf, id=id, division_type=DivisionType.EQUAL_DIVISION, term="DC")
