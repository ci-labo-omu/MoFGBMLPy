from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF


cdef class TriangularFuzzySet(FuzzySet):
    def __init__(self, left, center, right, id, term):
        super().__init__(function=TriangularMF(left, center, right), id=id, term=term)
