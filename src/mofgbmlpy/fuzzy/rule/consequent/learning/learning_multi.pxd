import numpy as np
import cython
cimport numpy as cnp

from mofgbmlpy.data.class_label.class_label_multi cimport ClassLabelMulti
from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.learning.abstract_learning cimport AbstractLearning
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi cimport RuleWeightMulti

cdef class LearningMulti(AbstractLearning):
    cpdef Consequent learning(self, Antecedent antecedent, Dataset dataset=?, double reject_threshold=?)
    cdef double[:,:] calc_confidence(self, Antecedent antecedent, Dataset dataset=?)
    cpdef ClassLabelMulti calc_class_label(self, double[:,:] confidence)
    cpdef RuleWeightMulti calc_rule_weight(self, ClassLabelMulti class_label, double[:,:] confidence, double reject_threshold)
    cpdef Dataset get_training_set(self)
