import copy

import numpy as np

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
cimport numpy as cnp


cdef class MichiganSolution(AbstractSolution):
    cdef object _bounds
    cdef object _rule
    cdef RuleBuilderCore _rule_builder

    cdef double get_lower_bound(self, int index)
    cdef double get_upper_bound(self, int index)
    cdef void create_rule(self, MichiganSolution michigan_solution=?, Pattern pattern=?)
    cpdef void learning(self)
    cpdef double get_fitness_value(self, cnp.ndarray[double, ndim=1] in_vector)
    cpdef int get_rule_length(self)
    cpdef get_class_label(self)
    cdef AbstractRuleWeight get_rule_weight(self)
    cdef object get_vars_array(self)
    cpdef AbstractRule get_rule(self)
    cdef RuleBuilderCore get_rule_builder(self)
    cpdef Consequent get_consequent(self)
    cpdef Antecedent get_antecedent(self)
    cdef cnp.ndarray[double, ndim=1] get_compatible_grade(self, cnp.ndarray[double, ndim=1] attribute_vector)
    cdef double get_compatible_grade_value(self, cnp.ndarray[double, ndim=1] attribute_vector)
    cpdef double compute_coverage(self)