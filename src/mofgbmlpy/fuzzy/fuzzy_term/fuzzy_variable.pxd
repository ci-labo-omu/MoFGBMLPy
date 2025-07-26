from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet, FuzzySetCpp

from libcpp.vector cimport vector
from libcpp.string cimport string as std_string


cdef extern from "core/fuzzy/fuzzy_term/fuzzy_variable.hpp":
    cdef cppclass FuzzyVariableCpp "FuzzyVariable":
        FuzzyVariableCpp(const vector[FuzzySetCpp*]& fuzzy_sets, const std_string& name, const vector[float]& domain) except +;
        FuzzyVariableCpp(const FuzzyVariableCpp& other) except +;
        std_string get_name() const;
        float get_membership_value(int fuzzy_set_index, float x) except +;
        int get_length() const;
        FuzzySetCpp* get_fuzzy_set(int fuzzy_set_index) except +;
        float get_support(int fuzzy_set_index) except +;
        vector[FuzzySetCpp*] get_fuzzy_sets() const;
        vector[float] get_support_values() const;
        vector[float] get_domain() const;
        FuzzyVariableCpp* clone() const;
        std_string to_string() const;
        bint operator==(const FuzzyVariableCpp& other) const;

cdef class FuzzyVariable:
    cdef FuzzyVariableCpp* ptr
    cpdef str get_name(self)
    cdef float get_membership_value(self, int fuzzy_set_index, float x)
    cpdef int get_length(self)
    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index)
    cpdef float get_support(self, int fuzzy_set_index)
    cpdef get_fuzzy_sets(self)
    cpdef get_support_values(self)
    cpdef get_domain(self)
    @staticmethod
    cdef FuzzyVariable wrap(FuzzyVariableCpp * ptr)