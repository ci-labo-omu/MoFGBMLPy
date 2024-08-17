from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore

cdef class SolutionBuilderCore:
    """Core builder for solutions

    Attributes:
        _num_objectives (int): Number of objectives
        _num_constraints (int): Number of constraints
        _rule_builder (RuleBuilderCore): Rule builder
    """
    def __init__(self, num_objectives, num_constraints, rule_builder):
        """Constructor

        Args:
            num_objectives (int): Number of objectives
            num_constraints (int): Number of constraints
            rule_builder (RuleBuilderCore): Rule builder
        """
        self._num_objectives = num_objectives
        self._num_constraints = num_constraints
        self._rule_builder = rule_builder

    cpdef RuleBuilderCore get_rule_builder(self):
        """Get the rule builder
        
        Returns:
            RuleBuilderCore: Rule builder
        """
        return self._rule_builder