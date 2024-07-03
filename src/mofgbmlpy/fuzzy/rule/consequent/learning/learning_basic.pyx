import numpy as np

cimport numpy as cnp

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.learning.abstract_learning cimport AbstractLearning
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic cimport RuleWeightBasic
from mofgbmlpy.data.pattern cimport Pattern
from libc.math cimport INFINITY
from cython.parallel import prange


cdef class LearningBasic(AbstractLearning):
    def __init__(self, training_dataset):
        self._train_ds = training_dataset

    cpdef Consequent learning(self, Antecedent antecedent, double reject_threshold=0):
        cdef double[:] confidence = self.calc_confidence(antecedent)
        cdef ClassLabelBasic class_label = self.calc_class_label(confidence)
        cdef RuleWeightBasic rule_weight = self.calc_rule_weight(class_label, confidence, reject_threshold)

        return Consequent(class_label, rule_weight)

    cdef double[:] calc_confidence(self, Antecedent antecedent):
        if antecedent is None:
            # with cython.gil:
            raise ValueError('Antecedent cannot be None')

        cdef int num_classes = self._train_ds.get_num_classes()
        cdef double[:] confidence = np.zeros(num_classes)
        cdef cnp.ndarray[double, ndim=1] sum_compatible_grade_for_each_class = np.zeros(num_classes)
        cdef double[:] compatible_grades = np.zeros(self._train_ds.get_size())
        cdef Pattern[:] patterns = self._train_ds.get_patterns()
        cdef int i
        cdef Pattern p

        # for i in prange(self._train_ds.get_size(), nogil=True):
        for i in range(self._train_ds.get_size()):
            p = patterns[i]
            compatible_grades[i] = antecedent.get_compatible_grade_value(p.get_attributes_vector())

        cdef double all_sum = 0
        cdef int c
        cdef double part_sum = 0
        cdef int class_label

        for c in range(num_classes):
            part_sum = 0
            # TODO: Add multithreading
            for i in range(self._train_ds.get_size()):
                pattern = patterns[i]
                if pattern.get_target_class().get_class_label_value() == c:
                    part_sum += compatible_grades[i]

            sum_compatible_grade_for_each_class[c] = part_sum
            all_sum += part_sum

        if all_sum != 0:
            confidence = sum_compatible_grade_for_each_class/all_sum

        return confidence

    cpdef ClassLabelBasic calc_class_label(self, double[:] confidence):
        cdef double max_val = -INFINITY
        cdef int consequent_class = -1
        cdef int i

        for i in range(confidence.shape[0]):
            if confidence[i] > max_val:
                max_val = confidence[i]
                consequent_class = i
            elif confidence[i] == max_val:
                consequent_class = -1

        if consequent_class < 0:
            class_label = ClassLabelBasic(-1)
            class_label.set_rejected()
        else:
            class_label = ClassLabelBasic(consequent_class)
        return class_label

    cpdef RuleWeightBasic calc_rule_weight(self, ClassLabelBasic class_label, double[:] confidence, double reject_threshold):
        cdef RuleWeightBasic zero_weight = RuleWeightBasic(0.0)

        if class_label.is_rejected():
            return zero_weight

        cdef int label_value = class_label.get_class_label_value()
        cdef double sum_confidence = np.sum(confidence)
        cdef double rule_weight_val = confidence[label_value] - (sum_confidence - confidence[label_value])

        if rule_weight_val <= reject_threshold:
            class_label.set_rejected() # TODO: check if it's modified globally or locally only
            return zero_weight

        return RuleWeightBasic(rule_weight_val)

    cpdef Dataset get_training_set(self):
        return self._train_ds

    def __str__(self):
        return f"MoFGBML_Learning [defaultLimit={AbstractLearning._default_reject_threshold}]"

    def __deepcopy__(self, memo={}):
        new_object = LearningBasic(self._train_ds)

        memo[id(self)] = new_object
        return new_object
