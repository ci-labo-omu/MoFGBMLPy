from pymoo.core.duplicate import ElementwiseDuplicateElimination


class BasicDuplicateElimination(ElementwiseDuplicateElimination):
    """Basic duplicate solutions elimination using a simple "==" test on the first variable (there is always only one variable for Pittsburgh solutions and Michigan solutions populations """
    def is_equal(self, a, b):
        """Check if 2 solutions are equal by checking the equality between their first variable

        Args:
            a (AbstractSolution):
            b (AbstractSolution):

        Returns:

        """
        return a.X[0] == b.X[0]
