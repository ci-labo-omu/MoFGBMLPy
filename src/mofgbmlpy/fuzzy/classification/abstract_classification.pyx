cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern
import cython
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class AbstractClassification:
    """Abstract class for classification methods. """
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern):
        raise Exception("AbstractClassification is abstract")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        raise Exception("AbstractClassification is abstract")
