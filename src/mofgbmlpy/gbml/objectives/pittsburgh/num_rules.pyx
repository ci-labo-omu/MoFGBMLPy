from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution cimport PittsburghSolution

cdef class NumRules(ObjectiveFunction):
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        cdef int i = 0
        cdef PittsburghSolution sol

        if isinstance(solutions[0], PittsburghSolution):
            for i in range(len(solutions)):
                sol = solutions[i]
                out[i] = sol.get_num_vars()
                sol.set_objective(obj_index, out[i])
        else:
            raise Exception("Solution must be of type PittsburghSolution")

    def __repr__(self):
        return "Number of rules"