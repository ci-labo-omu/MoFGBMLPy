from mofgbmlpy.gbml.objectives.objective_function cimport ObjectiveFunction
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution cimport PittsburghSolution
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType


cdef class RuleInterpretation(ObjectiveFunction):
    """Objective function that was in the original Java version. Its usage was not yet properly investigated so please refrain from using it"""
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        """Run the objective function on the given parameters

        Args:
            solutions (AbstractSolution[]): Solutions that are evaluated
            obj_index (int): Index of the objective in the solution objectives array
            out (double[]): Output array, it will contain the objective value of all the solutions
        """
        cdef int i = 0
        cdef PittsburghSolution sol
        cdef double rule_interpretation
        cdef MichiganSolution michigan_sol

        if isinstance(solutions[0], PittsburghSolution):
            for i in range(len(solutions)):
                sol = solutions[i]

                knowledge = sol.get_michigan_solution_builder().get_rule_builder().get_knowledge()
                rule_interpretation = 0
                for j in range(sol.get_num_vars()):
                    michigan_sol = sol.get_var(j)
                    if michigan_sol.get_num_wins() > 0:
                        rule_interpretation += 1

                    rule_interpretation += michigan_sol.get_length() * 1e-4

                    indices = michigan_sol.get_antecedent().get_antecedent_indices()
                    for k in range(len(indices)):
                        if knowledge.get_fuzzy_set(k, indices[k]).get_division_type() == DivisionType.ENTROPY_DIVISION:
                            rule_interpretation += 1e-8

                out[i] = rule_interpretation
                sol.set_objective(obj_index, out[i])
        else:
            raise Exception("Solution must be of type PittsburghSolution")

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "Rule interpretation"
