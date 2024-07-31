from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution

cdef class ObjectiveFunction:
    cpdef void run(self, AbstractSolution[:] solutions, int obj_index, double[:] out):
        raise Exception("Abstract class")

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        raise Exception("Abstract class")