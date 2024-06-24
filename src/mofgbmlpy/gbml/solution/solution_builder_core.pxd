from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore

cdef class SolutionBuilderCore:
    cdef object _bounds
    cdef int _num_objectives
    cdef int _num_constraints
    cdef RuleBuilderCore _rule_builder

    cdef RuleBuilderCore get_rule_builder(self)