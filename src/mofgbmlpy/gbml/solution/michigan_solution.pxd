import copy

import numpy as np
import cython
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
    cdef AbstractRule _rule
    cdef RuleBuilderCore _rule_builder
    cdef int[:] _vars
    cdef int __num_wins
    cdef int __fitness

    cdef void create_rule(self, Pattern pattern=?)
    cpdef void learning(self)
    cpdef double get_fitness_value(self, double[:] in_vector)
    cpdef int get_length(self)
    cpdef get_class_label(self)
    cdef AbstractRuleWeight get_rule_weight(self)
    cpdef AbstractRuleWeight get_rule_weight_py(self)
    cpdef AbstractRule get_rule(self)
    cpdef RuleBuilderCore get_rule_builder(self)
    cpdef Consequent get_consequent(self)
    cpdef Antecedent get_antecedent(self)
    cdef double[:] get_compatible_grade(self, double[:] attribute_vector)
    cdef double get_compatible_grade_value(self, double[:] attribute_vector)
    cpdef double compute_coverage(self)
    cpdef void reset_num_wins(self)
    cpdef void reset_fitness(self)
    cpdef void inc_num_wins(self)
    cpdef void inc_fitness(self)
    cpdef int get_num_wins(self)
    cpdef int get_fitness(self)
    cdef void clear_vars(self)
    cpdef int[:] get_vars(self)
    cpdef int get_var(self, int index)
    cpdef void set_var(self, int index, int value)
    cpdef void set_vars(self, int[:] new_vars)
    cpdef int get_num_vars(self)
    cpdef void set_antecedent_knowledge(self, Knowledge knowledge)