from mofgbmlpy.exception.invalid_solution_type_exception import InvalidSolutionTypeException
from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution cimport PittsburghSolution

cdef class TotalRuleLength(ObjectiveFunction):
    """Objective function that considers the total rule length (sum of all its rule length) to evaluate Pittsburgh solutions"""
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        """Run the objective function on the given parameters

        Args:
            solutions (AbstractSolution[]): Solutions that are evaluated
            obj_index (int): Index of the objective in the solution objectives array
            out (double[]): Output array, it will contain the objective value of all the solutions
        """
        cdef int i = 0
        cdef PittsburghSolution sol

        if isinstance(solutions[0], PittsburghSolution):
            for i in range(len(solutions)):
                sol = solutions[i]
                out[i] = sol.get_total_rule_length()
                sol.set_objective(obj_index, out[i])
        else:
            raise InvalidSolutionTypeException("PittsburghSolution")

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "Total rule length"