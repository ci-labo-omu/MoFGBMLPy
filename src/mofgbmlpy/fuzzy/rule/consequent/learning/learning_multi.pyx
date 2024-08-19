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
    def __init__(self, Dataset training_dataset):
        """Constructor

        Args:
            training_dataset (Dataset): Training dataset used to generate the consequent
        """
        super().__init__(training_dataset)

    cpdef Consequent learning(self, Antecedent antecedent, Dataset dataset=None, double reject_threshold=0):
        """Learn a consequent from the antecedent and dataset

        Args:
            antecedent (Antecedent): Antecedent whose consequent part is learnt
            dataset (Dataset): Training dataset
            reject_threshold (double): Threshold for the rule weight under which the rule is considered rejected

        Returns:
            Consequent: Created consequent
        """
        cdef double[:,:] confidence = self.calc_confidence(antecedent)
        cdef ClassLabelMulti class_label = self.calc_class_label(confidence)
        cdef RuleWeightMulti rule_weight = self.calc_rule_weight(class_label, confidence, reject_threshold)
        return Consequent(class_label, rule_weight)

    cdef double[:,:] calc_confidence(self, Antecedent antecedent, Dataset dataset=None):
        """Compute the confidences of each class for the given antecedent and dataset. Can only be accessed from Cython code

        Args:
            antecedent (Antecedent): Antecedent whose confidence is computed 
            dataset (Dataset): Training dataset

        Returns:
            double[,]: Confidence. e.g. confidence[0, 0] is the confidence that the class 0 is not i the multi class label, and confidence[0, 1] is the confidence that it is 
        """
        if dataset is None:
            dataset = self._train_ds
        if antecedent is None:
            raise TypeError('Antecedent cannot be None')

        cdef int num_classes = dataset.get_num_classes()
        cdef double[:,:] confidence = np.zeros((num_classes, 2))
        cdef double[:] compatible_grades = np.zeros(dataset.get_size())
        cdef Pattern[:] patterns = dataset.get_patterns()
        cdef int i
        cdef Pattern p

        # for i in prange(dataset.get_size(), nogil=True):
        for i in range(dataset.get_size()):
            p = patterns[i]
            compatible_grades[i] = antecedent.get_compatible_grade_value(p.get_attributes_vector())

        cdef double all_sum
        cdef int c
        cdef int class_label
        cdef int class_label_val

        for c in range(num_classes):
            # TODO: Add multithreading
            for i in range(dataset.get_size()):
                pattern = patterns[i]
                class_label_val = pattern.get_target_class().get_class_label_value_at(c)
                confidence[c][class_label_val] += compatible_grades[i]

            all_sum = confidence[c][0] + confidence[c][1]
            if all_sum != 0:
                confidence[c][0] /= all_sum
                confidence[c][1] /= all_sum
            else:
                confidence[c][0] = 0.0
                confidence[c][1] = 0.0

        return confidence

    cpdef double[:,:] calc_confidence_py(self, Antecedent antecedent, Dataset dataset=None):
        return self.calc_confidence(antecedent, dataset)

    cpdef ClassLabelMulti calc_class_label(self, double[:,:] confidence):
        """Compute the conclusion class label using the confidence
        
        Args:
            confidence (double[,]): confidences of each class for the given antecedent and dataset

        Returns:
            ClassLabelMulti: Label object containing a list of 0 and 1. 1 if the class is present and 0 otherwise . If the confidence that it is present and the confidence that is not are equal then the rule is rejected
        """
        cdef double max_val = -INFINITY
        cdef int[:] consequent_classes = np.full((confidence.shape[0]), fill_value=-1)
        cdef int c

        if confidence is None:
            raise TypeError("confidence can't be None")

        for c in range(confidence.shape[0]):
            if confidence[c][0] > confidence[c][1]:
                consequent_classes[c] = 0 # It's more likely "OFF" than "ON"
            elif confidence[c][0] < confidence[c][1]:
                consequent_classes[c] = 1
            else:
                class_label = ClassLabelMulti(consequent_classes)
                class_label.set_rejected()
                return class_label

        return ClassLabelMulti(consequent_classes)

    cpdef RuleWeightMulti calc_rule_weight(self, ClassLabelMulti class_label, double[:,:] confidence, double reject_threshold):
        """Compute the rule weight

        Args:
            class_label (ClassLabelMulti): Class label whose rule weight is computed
            confidence (double[,]): confidences of each class for the given antecedent and dataset
            reject_threshold (double): Threshold for the rule weight value under which the rule is considered rejected

        Returns:
            RuleWeightMulti: Rule weight
        """
        if confidence is None:
            raise TypeError("confidence can't be None")
        elif class_label is None:
            raise TypeError("class_label can't be None")

        cdef double[:] rule_weight_values = np.full((confidence.shape[0]), fill_value=-1.0)

        if not class_label.is_rejected():
            # TODO: use Numpy instead if possible
            for c in range(confidence.shape[0]):
                rule_weight_values[c] = abs(confidence[c][0] - confidence[c][1])

        return RuleWeightMulti(rule_weight_values)

    cpdef Dataset get_training_set(self):
        """Get the training set

        Returns:
            Dataset: Training set
        """
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

    def __eq__(self, other):
        """Check if another object is equal to this one

        Args:
            other (object): Object compared to this one

        Returns:
            bool: True if they are equal and False otherwise
        """
        if not isinstance(other, LearningMulti):
            return False

        return self._train_ds == other.get_training_set()