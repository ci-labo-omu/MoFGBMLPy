import numpy as np
import cython
cimport numpy as cnp

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.abstract_consequent cimport AbstractConsequent
from mofgbmlpy.fuzzy.rule.consequent.learning.abstract_learning cimport AbstractLearning
from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic cimport RuleWeightBasic
from mofgbmlpy.data.pattern cimport Pattern

cdef class LearningBasic(AbstractLearning):
    cpdef AbstractConsequent learning(self, Antecedent antecedent, Dataset dataset=?, float reject_threshold=?)
    cdef float[:] calc_confidence(self, Antecedent antecedent, Dataset dataset=?)
    cpdef float[:] calc_confidence_py(self, Antecedent antecedent, Dataset dataset=?)
    cpdef ClassLabelBasic calc_class_label(self, float[:] confidence)
    cpdef RuleWeightBasic calc_rule_weight(self, ClassLabelBasic class_label, float[:] confidence, float reject_threshold)
    cpdef Dataset get_training_set(self)
