from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge, KnowledgeCpp
cimport numpy as cnp
import cython


from libcpp.vector cimport vector
from libcpp.string cimport string as std_string


cdef extern from "core/fuzzy/rule/antecedent/antecedent.hpp":
    cdef cppclass AntecedentCpp "Antecedent":
        AntecedentCpp(vector[int]& antecedent_indices, KnowledgeCpp* knowledge) except +;
        AntecedentCpp(const AntecedentCpp& other) except +;
    
        int get_array_size() const;
        vector[int] get_antecedent_indices() const;
        void set_antecedent_indices(const vector[int]& new_indices) except +;
        vector[double] get_membership_values(const vector[double]& attribute_vector) except +;
        double get_compatible_grade_value(const vector[double]& attribute_vector) except +;
        int get_length() const;
        std_string get_linguistic_representation() const;
        KnowledgeCpp* get_knowledge() const;
        void set_knowledge(KnowledgeCpp* new_knowledge);
    
        std_string to_string() const;
        bint operator==(const AntecedentCpp& other) const;
        AntecedentCpp* clone() const;

cdef class Antecedent:
    cdef AntecedentCpp* ptr

    cpdef int get_array_size(self)
    cpdef int[:] get_antecedent_indices(self)
    cpdef void set_antecedent_indices(self, int[:] new_indices)
    cpdef double[:] get_membership_values(self, double[:] attribute_vector)
    cdef double get_compatible_grade_value(self, double[:] attribute_vector)
    cpdef int get_length(self)
    cpdef str get_linguistic_representation(self)
    cpdef get_knowledge(self)
    cpdef set_knowledge(self, Knowledge new_knowledge)
    @staticmethod
    cdef Antecedent wrap(AntecedentCpp * wrapped_ptr)