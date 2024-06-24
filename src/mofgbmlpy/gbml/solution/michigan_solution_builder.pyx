import copy

import numpy as np

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.gbml.solution.solution_builder_core cimport SolutionBuilderCore
cimport numpy as cnp

cdef class MichiganSolutionBuilder(SolutionBuilderCore):
    def __init__(self, bounds, num_objectives, num_constraints, rule_builder):
        super().__init__(bounds, num_objectives, num_constraints, rule_builder)

    cpdef cnp.ndarray[object, ndim=1] create(self, int num_solutions=1, Pattern pattern=None):
        cdef cnp.ndarray[object, ndim=1] solutions = np.empty(num_solutions, dtype=object)
        cdef int i
        bounds = self._bounds

        if bounds is None:
            bounds = MichiganSolution.make_bounds(self._rule_builder.get_knowledge())

        for i in range(num_solutions):
            solutions[i] = MichiganSolution(self._num_objectives, self._num_constraints, self._rule_builder, bounds,
                                            pattern=pattern)
            # solutions[i].set_attribute(attribute_id, 0) # TODO: check usage in java version
            # solutions[i].set_attribute(attribute_id_fitness, 0)

        return solutions

    def __copy__(self):
        return MichiganSolutionBuilder(self._bounds, self._num_objectives, self._num_constraints,
                                                        copy.copy(self._rule_builder))