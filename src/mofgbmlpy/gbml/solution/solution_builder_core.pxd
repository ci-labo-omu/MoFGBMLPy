from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore

cdef class SolutionBuilderCore:
    cdef int _num_objectives
    cdef int _num_constraints
    cdef RuleBuilderCore _rule_builder

    cpdef RuleBuilderCore get_rule_builder(self)