from pymoo.core.duplicate import ElementwiseDuplicateElimination


# Redefinition of this method is required so that pymoo works with custom types

class BasicDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return False
