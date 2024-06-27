from abc import ABC, abstractmethod

import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight

cdef class AbstractRule:
    def __init__(self, antecedent, consequent):
        self._antecedent = antecedent
        self._consequent = consequent

    cpdef Antecedent get_antecedent(self):
        return self._antecedent

    cpdef Consequent get_consequent(self):
        return self._consequent

    cpdef void set_consequent(self, Consequent consequent):
        self._consequent = consequent

    cdef cnp.ndarray[double, ndim=1] get_compatible_grade(self, cnp.ndarray[double, ndim=1] attribute_vector):
        return self._antecedent.get_compatible_grade(attribute_vector)

    cdef double get_compatible_grade_value(self, cnp.ndarray[double, ndim=1] attribute_vector):
        return self._antecedent.get_compatible_grade_value(attribute_vector)

    cpdef AbstractClassLabel get_class_label(self):
        return self._consequent.get_class_label()

    cpdef object get_class_label_value(self):
        return self._consequent.get_class_label_value()

    cdef cnp.npy_bool equals_class_label(self, AbstractRule other):
        return self._consequent().get_class_label() == other.get_consequent().get_class_label()

    cpdef cnp.npy_bool is_rejected_class_label(self):
        return self._consequent.get_class_label().is_rejected()

    cdef AbstractRuleWeight get_rule_weight(self):
        return self._consequent.get_rule_weight()

    cpdef AbstractRuleWeight get_rule_weight_py(self):
        return self._consequent.get_rule_weight()

    cdef object get_rule_weight_value(self):
        return self._consequent.get_rule_weight().get_value()

    cdef set_rule_weight_value(self, object rule_weight_value):
        self._consequent.set_rule_weight_value(rule_weight_value)

    cdef set_class_label_value(self, object class_label_value):
        self.get_consequent().set_class_label_value(class_label_value)

    cpdef int get_rule_length(self):
        return self.get_antecedent().get_length()

    cpdef double get_fitness_value(self, cnp.ndarray[double, ndim=1] attribute_vector):
        Exception("AbstractRule is abstract")

    def __eq__(self, other):
        return self._antecedent == other.get_antecedent() and self._consequent == other.get_consequent()

    def __str__(self):
        return f"Antecedent: {self._antecedent} => Consequent: {self._consequent}"