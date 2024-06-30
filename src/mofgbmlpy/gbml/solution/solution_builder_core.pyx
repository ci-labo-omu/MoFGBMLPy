from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore

cdef class SolutionBuilderCore:
    def __init__(self, bounds, num_objectives, num_constraints, rule_builder):
        self._bounds = bounds
        self._num_objectives = num_objectives
        self._num_constraints = num_constraints
        self._rule_builder = rule_builder

    cpdef RuleBuilderCore get_rule_builder(self):
        return self._rule_builder