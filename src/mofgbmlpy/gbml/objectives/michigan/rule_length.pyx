from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class RuleLength(ObjectiveFunction):
    """Objective function that considers the rule length to evaluate Michigan solutions"""
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        """Run the objective function on the given parameters

        Args:
            solutions (MichiganSolution[]): Solutions that are evaluated
            obj_index (int): Index of the objective in the solution objectives array
            out (double[]): Output array, it will contain the objective value of all the solutions
        """
        cdef int i = 0
        cdef MichiganSolution sol

        if isinstance(solutions[0], MichiganSolution):
            for i in range(len(solutions)):
                sol = solutions[i]
                out[i] = sol.get_length()
                sol.set_objective(obj_index, out[i])
        else:
            raise Exception("Solution must be of type MichiganSolution")

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "Rule length"