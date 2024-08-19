from mofgbmlpy.exception.abstract_class_exception import AbstractMethodException

cdef class AbstractRuleWeight:
    """Abstract class for rule weights object"""
    cpdef object get_value(self):
        """Get the rule weight value
        
        Returns:
            object: Rule weight value
        """
        raise AbstractMethodException()

    cpdef void set_value(self, object rule_weight):
        """Set the value of the rule weight
        
        Args:
            rule_weight (object): New rule weight value
        """
        raise AbstractMethodException()
