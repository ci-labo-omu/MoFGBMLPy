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
    def __init__(self, Dataset training_dataset):
        if training_dataset is None:
            raise Exception("The training dataset cannot be None")
        self._train_ds = training_dataset

    cpdef Consequent learning(self, Antecedent antecedent, Dataset dataset=None, double reject_threshold=0):
        cdef double[:] confidence = self.calc_confidence(antecedent)
        cdef ClassLabelBasic class_label = self.calc_class_label(confidence)
        cdef RuleWeightBasic rule_weight = self.calc_rule_weight(class_label, confidence, reject_threshold)
        return Consequent(class_label, rule_weight)

    cdef double[:] calc_confidence(self, Antecedent antecedent, Dataset dataset=None):
        if dataset is None:
            dataset = self._train_ds
        if antecedent is None:
            raise ValueError('Antecedent cannot be None')

        cdef int num_classes = dataset.get_num_classes()
        cdef double[:] confidence = np.zeros(num_classes)
        cdef cnp.ndarray[double, ndim=1] sum_compatible_grade_for_each_class = np.zeros(num_classes)
        cdef double[:] compatible_grades = np.zeros(dataset.get_size())
        cdef Pattern[:] patterns = dataset.get_patterns()
        cdef int i
        cdef Pattern p

        # for i in prange(dataset.get_size(), nogil=True):
        for i in range(dataset.get_size()):
            p = patterns[i]
            compatible_grades[i] = antecedent.get_compatible_grade_value(p.get_attributes_vector())

        cdef double all_sum = 0
        cdef int c
        cdef double part_sum = 0
        cdef int class_label

        for c in range(num_classes):
            part_sum = 0
            # TODO: Add multithreading
            for i in range(dataset.get_size()):
                pattern = patterns[i]
                if pattern.get_target_class().get_class_label_value() == c:
                    part_sum += compatible_grades[i]

            sum_compatible_grade_for_each_class[c] = part_sum
            all_sum += part_sum

        if all_sum != 0:
            confidence = sum_compatible_grade_for_each_class/all_sum

        return confidence

    cpdef double[:] calc_confidence_py(self, Antecedent antecedent):
        return self.calc_confidence(antecedent)


    cpdef ClassLabelBasic calc_class_label(self, double[:] confidence):
        cdef double max_val = -INFINITY
        cdef int consequent_class = -1
        cdef int i

        if confidence is None:
            raise Exception("confidence can't be None")

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
        if class_label is None or confidence is None or reject_threshold is None:
            raise Exception("Class label, confidence and reject_threshold can't be None")

        cdef RuleWeightBasic zero_weight = RuleWeightBasic(0.0)

        if class_label.is_rejected():
            return zero_weight

        cdef int label_value = class_label.get_class_label_value()

        # TODO Check the effect of this modification on the results and recheck it's validity
        # cdef double sum_confidence = np.sum(confidence)
        # cdef double rule_weight_val = confidence[label_value] - (sum_confidence - confidence[label_value])
        cdef double rule_weight_val = (confidence[label_value] * 2) - 1


        if rule_weight_val <= reject_threshold:
            class_label.set_rejected()
            return zero_weight

        return RuleWeightBasic(rule_weight_val)

    cpdef Dataset get_training_set(self):
        return self._train_ds

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"MoFGBML_Learning"

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object.
        Note that the dataset is the same object (not deep copied)

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = LearningBasic(self._train_ds)

        memo[id(self)] = new_object
        return new_object

    def __eq__(self, other):
        """Check if another object is equal to this one

        Args:
            other (object): Object compared to this one

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, LearningBasic):
            return False

        return self._train_ds == other.get_training_set()
