from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySetCpp, FuzzySet

cdef extern from "core/fuzzy/fuzzy_term/fuzzy_set/dont_care_fuzzy_set.hpp":
    cdef cppclass DontCareFuzzySetCpp "DontCareFuzzySet"(FuzzySetCpp):
        DontCareFuzzySetCpp(int id);
        DontCareFuzzySetCpp(const DontCareFuzzySetCpp& other);

cdef class DontCareFuzzySet(FuzzySet):
    pass
