from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySetCpp, FuzzySet
from libcpp.string cimport string as std_string

cdef extern from "core/fuzzy/fuzzy_term/fuzzy_set/rectangular_fuzzy_set.hpp":
    cdef cppclass RectangularFuzzySetCpp "RectangularFuzzySet"(FuzzySetCpp):
        RectangularFuzzySetCpp(float left, float right, int id, const std_string& term);
        RectangularFuzzySetCpp(const RectangularFuzzySetCpp& other);

cdef class RectangularFuzzySet(FuzzySet):
    pass
