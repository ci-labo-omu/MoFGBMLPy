from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.abstract_consequent cimport AbstractConsequent
from mofgbmlpy.data.dataset_density cimport DatasetWithDensity
import cython
from mofgbmlpy.data.dataset_manager cimport DatasetManager

cdef class AbstractLearningWithDensity:
    cdef DatasetManager dataset_manager
    cdef DatasetWithDensity __train_ds

    cpdef AbstractConsequent learning(self, Antecedent antecedent, DatasetWithDensity dataset=?, double reject_threshold=?)
