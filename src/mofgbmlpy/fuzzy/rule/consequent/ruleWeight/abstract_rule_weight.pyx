cimport numpy as cnp


cdef class AbstractRuleWeight:
    """Abstract class for rule weights object"""
    cpdef object get_value(self):
        """Get the rule weight value
        
        Returns:
            object: Rule weight value
        """
        raise Exception("AbstractRuleWeight is abstract")

    cpdef void set_value(self, object rule_weight):
        """Set the value of the rule weight
        
        Args:
            rule_weight (object): New rule weight value
        """
        raise Exception("AbstractRuleWeight is abstract")
