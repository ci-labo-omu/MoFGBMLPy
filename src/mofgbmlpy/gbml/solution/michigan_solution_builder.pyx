import copy

import numpy as np

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
from mofgbmlpy.gbml.solution.solution_builder_core cimport SolutionBuilderCore
cimport numpy as cnp

cdef class MichiganSolutionBuilder(SolutionBuilderCore):
    def __init__(self, num_objectives, num_constraints, rule_builder):
        super().__init__(num_objectives, num_constraints, rule_builder)

    cpdef MichiganSolution[:] create(self, int num_solutions=1, Pattern pattern=None):
        cdef MichiganSolution[:] solutions = np.empty(num_solutions, dtype=object)
        cdef int i

        for i in range(num_solutions):
            solutions[i] = MichiganSolution(self._num_objectives, self._num_constraints, self._rule_builder,
                                            pattern=pattern)

        return solutions

    def __deepcopy__(self, memo={}):
        cdef int i

        new_object = MichiganSolutionBuilder(self._num_objectives,
                                             self._num_constraints,
                                             copy.deepcopy(self._rule_builder))

        memo[id(self)] = new_object

        return new_object