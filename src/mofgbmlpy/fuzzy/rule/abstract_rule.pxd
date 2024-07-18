from abc import ABC, abstractmethod

import numpy as np
cimport numpy as cnp

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
import cython


cdef class AbstractRule:
    cdef Antecedent _antecedent
    cdef Consequent _consequent

    cpdef Antecedent get_antecedent(self)
    cpdef Consequent get_consequent(self)
    cpdef void set_consequent(self, Consequent consequent)
    cdef double[:] get_compatible_grade(self, double[:] attribute_vector)
    cdef double get_compatible_grade_value(self, double[:] attribute_vector)
    cpdef AbstractClassLabel get_class_label(self)
    cpdef object get_class_label_value(self)
    cdef bint equals_class_label(self, AbstractRule other)
    cpdef bint is_rejected_class_label(self)
    cdef AbstractRuleWeight get_rule_weight(self)
    cpdef AbstractRuleWeight get_rule_weight_py(self)
    cdef object get_rule_weight_value(self)
    cdef set_rule_weight_value(self, object rule_weight_value)
    cdef set_class_label_value(self, object class_label_value)
    cpdef int get_length(self)
    cpdef double get_fitness_value(self, double[:] attribute_vector)
    cpdef Knowledge get_knowledge(self)
    cpdef FuzzySet get_fuzzy_set_object(self, int dim_index)
    cpdef int get_antecedent_array_size(self)
    cpdef str get_var_concept(self, int dim_index)
