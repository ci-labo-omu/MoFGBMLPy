import numpy as np

cimport numpy as cnp

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent_basic cimport ConsequentBasic
from mofgbmlpy.fuzzy.rule.consequent.learning.abstract_learning cimport AbstractLearning
from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic cimport RuleWeightBasic
from mofgbmlpy.data.pattern cimport Pattern
from libc.math cimport INFINITY
from cython.parallel import prange


cdef class LearningBasic(AbstractLearning):
    def __init__(self, Dataset training_dataset):
        """Constructor

        Args:
            training_dataset (Dataset): Training dataset used to generate the consequent
        """
        super().__init__(training_dataset)

    cpdef AbstractConsequent learning(self, Antecedent antecedent, Dataset dataset=None, float reject_threshold=0):
        """Learn a consequent from the antecedent and dataset

        Args:
            antecedent (Antecedent): Antecedent whose consequent part is learnt
            dataset (Dataset): Training dataset
            reject_threshold (float): Threshold for the rule weight under which the rule is considered rejected

        Returns:
            AbstractConsequent: Created consequent
        """
        cdef float[:] confidence = self.calc_confidence(antecedent, dataset)
        cdef ClassLabelBasic class_label = self.calc_class_label(confidence)
        cdef RuleWeightBasic rule_weight = self.calc_rule_weight(class_label, confidence, reject_threshold)
        return ConsequentBasic(class_label, rule_weight)

    cdef float[:] calc_confidence(self, Antecedent antecedent, Dataset dataset=None):
        """Compute the confidences of each class for the given antecedent and dataset. Can only be accessed from Cython code
        
        Args:
            antecedent (Antecedent): Antecedent whose confidence is computed 
            dataset (Dataset): Training dataset

        Returns:
            float[]: Confidence
        """
        if dataset is None:
            dataset = self._train_ds
        if antecedent is None:
            raise TypeError('Antecedent cannot be None')

        cdef int num_classes = dataset.get_num_classes()
        cdef float[:] confidence = np.zeros(num_classes, dtype=np.float32)
        cdef cnp.ndarray[float, ndim=1] sum_compatible_grade_for_each_class = np.zeros(num_classes, dtype=np.float32)
        cdef float[:] compatible_grades = np.zeros(dataset.get_size(), dtype=np.float32)
        cdef Pattern[:] patterns = dataset.get_patterns()
        cdef int i
        cdef Pattern p

        # for i in prange(dataset.get_size(), nogil=True):
        for i in range(dataset.get_size()):
            p = patterns[i]
            compatible_grades[i] = antecedent.get_compatible_grade_value(p.get_attributes_vector())

        cdef float all_sum = 0
        cdef int c
        cdef float part_sum = 0
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

    cpdef float[:] calc_confidence_py(self, Antecedent antecedent, Dataset dataset=None):
        """Compute the confidences of each class for the given antecedent and dataset
        
        Args:
            antecedent (Antecedent): Antecedent whose confidence is computed 
            dataset (Dataset): Training dataset

        Returns:
            float[]: Confidence
        """
        return self.calc_confidence(antecedent, dataset)


    cpdef ClassLabelBasic calc_class_label(self, float[:] confidence):
        """Compute the conclusion class label using the confidence
        
        Args:
            confidence (float[]): confidences of each class for the given antecedent and dataset

        Returns:
            ClassLabelBasic: Class label of the class with the highest confidence. If there are multiple ones or if there is none then a rejected class label is returned
        """
        cdef float max_val = -INFINITY
        cdef int consequent_class = -1
        cdef int i

        if confidence is None:
            raise TypeError("confidence can't be None")

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

    cpdef RuleWeightBasic calc_rule_weight(self, ClassLabelBasic class_label, float[:] confidence, float reject_threshold):
        """Compute the rule weight
        
        Args:
            class_label (ClassLabelBasic): Class label whose rule weight is computed
            confidence (float[]): confidences of each class for the given antecedent and dataset
            reject_threshold (float): Threshold for the rule weight value under which the rule is considered rejected

        Returns:
            RuleWeightBasic: Rule weight
        """
        if class_label is None or confidence is None or reject_threshold is None:
            raise TypeError("Class label, confidence and reject_threshold can't be None")

        cdef RuleWeightBasic zero_weight = RuleWeightBasic(0.0)

        if class_label.is_rejected():
            return zero_weight

        cdef int label_value = class_label.get_class_label_value()

        if label_value < 0 or label_value >= len(confidence):
            raise IndexError("Label value is out of bounds for the confidence array")

        # TODO Re-check the effect of this modification on the results and recheck it's validity
        # cdef float sum_confidence = np.sum(confidence)
        # cdef float rule_weight_val = confidence[label_value] - (sum_confidence - confidence[label_value])
        cdef float rule_weight_val = (confidence[label_value] * 2) - 1


        if rule_weight_val <= reject_threshold:
            class_label.set_rejected()
            return zero_weight

        return RuleWeightBasic(rule_weight_val)

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
