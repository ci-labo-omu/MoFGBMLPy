import numpy as np
import cython
cimport numpy as cnp

from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.abstract_consequent cimport AbstractConsequent
from mofgbmlpy.fuzzy.rule.consequent.learning.abstract_learning_density cimport AbstractLearningWithDensity
from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic cimport RuleWeightBasic
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.data.dataset_density cimport DatasetWithDensity

cdef class LearningBasicDensity(AbstractLearningWithDensity):
    cpdef AbstractConsequent learning(self, Antecedent antecedent, DatasetWithDensity dataset=?, double reject_threshold=?)
    cdef double[:] calc_confidence(self, Antecedent antecedent, DatasetWithDensity dataset=?)
    cpdef double[:] calc_confidence_py(self, Antecedent antecedent, DatasetWithDensity dataset=?)
    cpdef ClassLabelBasic calc_class_label(self, double[:] confidence)
    cpdef RuleWeightBasic calc_rule_weight(self, ClassLabelBasic class_label, double[:] confidence, double reject_threshold)
    cpdef DatasetWithDensity get_training_set(self)
