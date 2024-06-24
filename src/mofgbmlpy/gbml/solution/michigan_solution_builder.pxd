import copy

import numpy as np

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.gbml.solution.solution_builder_core cimport SolutionBuilderCore
cimport numpy as cnp

cdef class MichiganSolutionBuilder(SolutionBuilderCore):
    cpdef cnp.ndarray[object, ndim=1] create(self, int num_solutions=?, Pattern pattern=?)
