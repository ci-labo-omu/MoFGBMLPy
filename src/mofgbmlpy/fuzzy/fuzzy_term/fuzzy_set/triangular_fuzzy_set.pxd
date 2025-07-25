from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySetCpp, FuzzySet
from libcpp.string cimport string as std_string


cdef extern from "core/fuzzy/fuzzy_term/fuzzy_set/triangular_fuzzy_set.hpp":
    cdef cppclass TriangularFuzzySetCpp "TriangularFuzzySet"(FuzzySetCpp):
        TriangularFuzzySetCpp(float left, float center, float right, int id, const std_string& term);
        TriangularFuzzySetCpp(const TriangularFuzzySetCpp& other);

cdef class TriangularFuzzySet(FuzzySet):
    pass
