from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent



cdef class AbstractLearning:
    cpdef Consequent learning(self, Antecedent antecedent, double reject_threshold=0):
        Exception("This class is abstract")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        Exception("This class is abstract")
