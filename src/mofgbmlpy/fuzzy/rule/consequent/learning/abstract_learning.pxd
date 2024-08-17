from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
import cython

cdef class AbstractLearning:
    cdef Dataset _train_ds

    cpdef Consequent learning(self, Antecedent antecedent, Dataset dataset=?, double reject_threshold=?)
