from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution

cdef class ObjectiveFunction:
    """Objective function for optimization problems """
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        """Run the objective function on the given parameters
        
        Args:
            solutions (AbstractSolution[]): Solutions that are evaluated
            obj_index (int): Index of the objective in the solution objectives array
            out (double[]): Output array, it will contain the objective value of all the solutions
        """
        raise Exception("Abstract class")

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        raise Exception("Abstract class")