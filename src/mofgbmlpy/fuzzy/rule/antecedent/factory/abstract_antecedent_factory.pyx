from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent


cdef class AbstractAntecedentFactory:
    """Abstract factory for antecedent"""
    cdef Antecedent[:] create(self, int num_rules=1):
        """Create antecedents. Can only be accessed from Cython code
        
        Args:
            num_rules (int): Number of antecedent to be generated

        Returns:
            Antecedent[]: Generated antecedents
        """
        Exception("This class is abstract")

    cdef int[:,:] create_antecedent_indices(self, int num_rules=1):
        """Create antecedents indices. Can only be accessed from Cython code

        Args:
            num_rules (int): Number of antecedents indices arrays to be generated

        Returns:
            int[,]: Generated antecedents indices
        """
        Exception("This class is abstract")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        Exception("This class is abstract")
