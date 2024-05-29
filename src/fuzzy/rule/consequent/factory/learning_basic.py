import numpy as np

from src.fuzzy.rule.consequent.factory.abstract_learning import  AbstractLearning
from src.fuzzy.rule.consequent.consequent import Consequent
from src.fuzzy.rule.consequent.classLabel.class_label_basic import ClassLabelBasic

class LearningBasic(AbstractLearning):
    _train_ds = None

    def __init__(self, training_dataset):
        self._train_ds = training_dataset

    def learning(self, antecedent, antecedent_indices, reject_threshold=AbstractLearning._default_reject_threshold):
        confidence = self.calc_confidence(antecedent, antecedent_indices)
        class_label = self.calc_class_label(confidence)
        rule_weight = self.calc_rule_weight(class_label, confidence, reject_threshold)

        return Consequent(class_label, rule_weight)

    def calc_confidence(self, antecedent, antecedent_indices):
        if antecedent is None:
            raise ValueError('Antecedent cannot be None')

        num_classes = self._train_ds.get_num_classes()
        confidence = np.zeros(num_classes)
        sum_compatible_grade_for_each_class = np.zeros(num_classes)

        for c in range(num_classes):
            part_sum = None
            # TODO: Add multithreading

            for pattern in self._train_ds.get_patterns():
                temp = antecedent.get_compatible_grade_value(antecedent_indices, pattern.get_attribute_vector())
                if pattern.get_target_class() == c:
                    part_sum += temp

            sum_compatible_grade_for_each_class[c] = part_sum

        all_sum = np.sum(sum_compatible_grade_for_each_class)

        if all_sum != 0:
            confidence = sum_compatible_grade_for_each_class/all_sum

        return confidence

    def calc_class_label(self, confidence):
        max_val = float('-inf')
        consequent_class = -1

        for i in range(len(confidence)):
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

    def calc_rule_weight(self, class_label, confidence, reject_threshold): #TODO

    def __str__(self):
        return f"MoFGBML_Learning [defaultLimit={AbstractLearning._default_reject_threshold}]"

    def copy(self):
        return LearningBasic(self._train_ds)