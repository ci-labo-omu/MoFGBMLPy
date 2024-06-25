import numpy as np
import cython
cimport numpy as cnp

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.learning.abstract_learning cimport AbstractLearning
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic cimport RuleWeightBasic
from mofgbmlpy.data.pattern cimport Pattern

cdef class LearningBasic(AbstractLearning):
    cdef Dataset _train_ds

    cpdef Consequent learning(self, Antecedent antecedent, double reject_threshold=?)
    cdef cnp.ndarray[double, ndim=1] calc_confidence(self, Antecedent antecedent)
    cpdef ClassLabelBasic calc_class_label(self, cnp.ndarray[double, ndim=1] confidence)
    cpdef RuleWeightBasic calc_rule_weight(self, ClassLabelBasic class_label, cnp.ndarray[double, ndim=1] confidence, double reject_threshold)
    cpdef Dataset get_training_set(self)
