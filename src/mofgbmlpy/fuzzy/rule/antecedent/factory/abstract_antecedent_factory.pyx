from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent


cdef class AbstractAntecedentFactory:
    cdef Antecedent[:] create(self, int num_rules=1):
        Exception("This class is abstract")

    cdef int[:,:] create_antecedent_indices(self, int num_rules=1):
        Exception("This class is abstract")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        Exception("This class is abstract")
