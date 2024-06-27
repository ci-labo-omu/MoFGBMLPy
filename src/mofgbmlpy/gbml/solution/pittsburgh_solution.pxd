import time

import numpy as np
cimport numpy as cnp
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classifier cimport Classifier
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution_builder cimport MichiganSolutionBuilder

cdef class PittsburghSolution(AbstractSolution):
    cdef Classifier __classifier
    cdef MichiganSolutionBuilder __michigan_solution_builder
    cdef object __errored_patterns

    cdef MichiganSolutionBuilder get_michigan_solution_builder(self)
    cpdef void remove_var(self, int index)
    cpdef void clear_vars(self)
    cpdef void clear_attributes(self)
    cdef void learning(self)
    cdef AbstractSolution classify(self, Pattern pattern)
    cpdef double get_error_rate(self, dataset)
    cdef cnp.ndarray[object, ndim=1] get_errored_patterns(self)
    cpdef double compute_coverage(self)
    cpdef int get_total_rule_weight(self)
