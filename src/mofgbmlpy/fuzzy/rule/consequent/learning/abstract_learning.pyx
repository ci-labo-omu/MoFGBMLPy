from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent



cdef class AbstractLearning:
    cpdef Consequent learning(self, Antecedent antecedent, double reject_threshold=0):
        Exception("This class is abstract")


    def __copy__(self):
        Exception("This class is abstract")
