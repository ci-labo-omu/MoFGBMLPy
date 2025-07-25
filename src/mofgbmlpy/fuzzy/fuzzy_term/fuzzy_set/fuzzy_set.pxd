from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF, AbstractMFCpp

from libcpp.string cimport string as std_string
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type cimport DivisionTypeCpp


cdef extern from "core/fuzzy/fuzzy_term/fuzzy_set/fuzzy_set.hpp":
    cdef cppclass FuzzySetCpp "FuzzySet":
        FuzzySetCpp(AbstractMFCpp* function, int id, DivisionTypeCpp division_type, std_string term);
        FuzzySetCpp(const FuzzySetCpp& other);
        std_string to_string() const;

        AbstractMFCpp* get_function() const;
        void set_function(AbstractMFCpp* new_function);
        std_string get_term() const;
        int get_id() const;
        int get_division_type() const;

        float get_membership_value(float x) const;
        float get_support(float x_min, float x_max) const;

        bint operator==(const FuzzySetCpp& other) const;
        FuzzySetCpp* clone() const;

cdef class FuzzySet:
    cdef FuzzySetCpp* ptr

    cdef float get_membership_value(self, float x)
    cpdef get_term(self)
    cpdef get_function_callable(self)
    cpdef int get_id(self)
    cpdef AbstractMF get_function(self)
    cpdef set_function(self, AbstractMF function)
    cpdef get_division_type(self)
    cpdef float get_support(self, float x_min=?, float x_max=?)
    @staticmethod
    cdef FuzzySet wrap(FuzzySetCpp * wrapped_ptr)