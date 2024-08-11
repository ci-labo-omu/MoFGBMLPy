import numpy as np

cimport numpy as cnp

from mofgbmlpy.data.class_label.abstract_class_label import AbstractClassLabel
from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.learning.abstract_learning cimport AbstractLearning
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.data.pattern cimport Pattern
from libc.math cimport INFINITY


cdef class LearningMulti(AbstractLearning):
    def __init__(self, training_dataset):
        self._train_ds = training_dataset

    cpdef Consequent learning(self, Antecedent antecedent, double reject_threshold=0):
        cdef double[:,:] confidence = self.calc_confidence(antecedent)
        cdef ClassLabelMulti class_label = self.calc_class_label(confidence)
        cdef RuleWeightMulti rule_weight = self.calc_rule_weight(class_label, confidence, reject_threshold)
        return Consequent(class_label, rule_weight)

    cdef double[:,:] calc_confidence(self, Antecedent antecedent):
        if antecedent is None:
            # with cython.gil:
            raise ValueError('Antecedent cannot be None')

        cdef int num_classes = self._train_ds.get_num_classes()
        cdef double[:,:] confidence = np.zeros((num_classes, 2))
        cdef double[:] compatible_grades = np.zeros(self._train_ds.get_size())
        cdef Pattern[:] patterns = self._train_ds.get_patterns()
        cdef int i
        cdef Pattern p

        # for i in prange(self._train_ds.get_size(), nogil=True):
        for i in range(self._train_ds.get_size()):
            p = patterns[i]
            compatible_grades[i] = antecedent.get_compatible_grade_value(p.get_attributes_vector())

        cdef double all_sum
        cdef int c
        cdef int class_label
        cdef int class_label_val

        for c in range(num_classes):
            # TODO: Add multithreading
            for i in range(self._train_ds.get_size()):
                pattern = patterns[i]
                class_label_val = pattern.get_target_class().get_class_label_value_at(c)
                confidence[c][class_label_val] += 1

            all_sum = confidence[c][0] + confidence[c][1]
            if all_sum != 0:
                confidence[c][0] /= all_sum
                confidence[c][1] /= all_sum
            else:
                confidence[c][0] = 0.0
                confidence[c][1] = 0.0

        return confidence

    cpdef ClassLabelMulti calc_class_label(self, double[:,:] confidence):
        cdef double max_val = -INFINITY
        cdef int[:] consequent_classes = np.full((confidence.shape[0]), fill_value=-1)
        cdef int c

        for c in range(confidence.shape[0]):
            if confidence[c][0] > confidence[c][1]:
                consequent_classes[c] = 0 # It's more likely "ON" than "OFF"
            elif confidence[c][0] > confidence[c][1]:
                consequent_classes[c] = 1
            else:
                class_label = ClassLabelMulti(consequent_classes)
                class_label.set_rejected()
                return class_label

        return ClassLabelMulti(consequent_classes)

    cpdef RuleWeightMulti calc_rule_weight(self, ClassLabelMulti class_label, double[:,:] confidence, double reject_threshold):
        cdef double[:] rule_weight_values = np.full((confidence.shape[0]), fill_value=-1.0)

        if not class_label.is_rejected():
            # TODO: use Numpy instead if possible
            for c in range(confidence.shape[0]):
                rule_weight_values[c] = abs(confidence[c][0] - confidence[c][1])

        return RuleWeightMulti(rule_weight_values)

    cpdef Dataset get_training_set(self):
        return self._train_ds

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"MoFGBML_Learning"

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = LearningMulti(self._train_ds)

        memo[id(self)] = new_object
        return new_object
