from pymoo.core.duplicate import ElementwiseDuplicateElimination


class BasicDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0]
