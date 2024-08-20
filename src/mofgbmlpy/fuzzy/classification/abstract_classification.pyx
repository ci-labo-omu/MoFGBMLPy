from mofgbmlpy.exception.abstract_method_exception import AbstractMethodException
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class AbstractClassification:
    """Abstract class for classification methods. """
    cpdef MichiganSolution classify(self, MichiganSolution[:] michigan_solution_list, Pattern pattern):
        raise AbstractMethodException()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        raise AbstractMethodException()
