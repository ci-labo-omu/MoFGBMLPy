import math

from matplotlib import pyplot as plt
cimport numpy as cnp
import cython
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet #, FuzzySetCpp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable #, FuzzyVariableCpp
#
# from libcpp.vector cimport vector
# from libcpp.string cimport string as std_string
#
#
# cdef extern from "core/fuzzy/knowledge/knowledge.hpp":
#     cdef cppclass KnowledgeCpp "Knowledge":
#         KnowledgeCpp();
#         KnowledgeCpp(const vector[FuzzyVariable*]& fuzzy_vars);
#         KnowledgeCpp(const KnowledgeCpp& other);
#
#         FuzzyVariableCpp* get_fuzzy_variable(int dim) const;
#         FuzzySetCpp* get_fuzzy_set(int dim, int fuzzy_set_index) const;
#         int get_num_fuzzy_sets(int dim) const;
#         void set_fuzzy_vars(const vector[FuzzyVariableCpp*]& fuzzy_vars);
#         const vector[FuzzyVariableCpp*]& get_fuzzy_vars() const;
#         float get_membership_value(double attribute_value, int dim, int fuzzy_set_index) const;
#         int get_num_dim() const;
#         float get_support(int dim, int fuzzy_set_index) const;
#         KnowledgeCpp* clone() const;
#         std_string to_string() const;
#         bool operator==(const KnowledgeCpp& other) const;

cdef class Knowledge:
    # cdef KnowledgeCpp* ptr
    cdef FuzzyVariable[:] __fuzzy_vars

    cpdef FuzzyVariable get_fuzzy_variable(self, int dim)
    cpdef FuzzySet get_fuzzy_set(self, int dim, int fuzzy_set_id)
    cpdef int get_num_fuzzy_sets(self, int dim)
    cpdef void set_fuzzy_vars(self, FuzzyVariable[:] fuzzy_vars)
    cpdef FuzzyVariable[:] get_fuzzy_vars(self)
    cdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_index)
    cpdef double get_membership_value_py(self, double attribute_value, int dim, int fuzzy_set_index)
    cpdef int get_num_dim(self)
    cpdef double get_support(self, int dim, int fuzzy_set_index)