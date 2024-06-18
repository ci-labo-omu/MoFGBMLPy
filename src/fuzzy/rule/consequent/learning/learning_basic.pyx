import numpy as np

from fuzzy.rule.consequent.learning.abstract_learning import  AbstractLearning
from fuzzy.rule.consequent.consequent import Consequent
from data.class_label.class_label_basic import ClassLabelBasic
from fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic


class LearningBasic(AbstractLearning):
    _train_ds = None

    def __init__(self, training_dataset):
        self._train_ds = training_dataset

    def learning(self, antecedent, reject_threshold=AbstractLearning._default_reject_threshold):
        confidence = self.calc_confidence(antecedent)
        class_label = self.calc_class_label(confidence)
        rule_weight = self.calc_rule_weight(class_label, confidence, reject_threshold)

        return Consequent(class_label, rule_weight)

    def calc_confidence(self, antecedent):
        if antecedent is None:
            raise ValueError('Antecedent cannot be None')

        num_classes = self._train_ds.get_num_classes()
        confidence = np.zeros(num_classes)
        sum_compatible_grade_for_each_class = np.zeros(num_classes)

        compatible_grades = np.zeros(self._train_ds.get_size(), dtype=int)

        patterns = self._train_ds.get_patterns()
        for i in range(self._train_ds.get_size()):
            compatible_grades[i] = antecedent.get_compatible_grade_value(patterns[i].get_attributes_vector())

        all_sum = 0
        for c in range(num_classes):
            part_sum = 0
            # TODO: Add multithreading
            for i in range(self._train_ds.get_size()):
                if patterns[i].get_target_class().get_class_label_value() == c:
                    part_sum += compatible_grades[i]

            sum_compatible_grade_for_each_class[c] = part_sum
            all_sum += part_sum

        if all_sum != 0:
            confidence = sum_compatible_grade_for_each_class/all_sum

        return confidence

    def calc_class_label(self, confidence):
        max_val = float('-inf')
        consequent_class = -1

        for i in range(len(confidence)):
            # print(confidence[i], max_val)
            if confidence[i] > max_val:
                max_val = confidence[i]
                consequent_class = i
            elif confidence[i] == max_val:
                consequent_class = -1

        # print("\n####################\n\n")

        if consequent_class < 0:
            class_label = ClassLabelBasic(-1)
            class_label.set_rejected()
        else:
            class_label = ClassLabelBasic(consequent_class)
        return class_label

    def calc_rule_weight(self, class_label, confidence, reject_threshold):
        zero_weight = RuleWeightBasic(0.0)

        if class_label.is_rejected():
            return zero_weight

        label_value = class_label.get_class_label_value()
        sum_confidence = np.sum(confidence)
        rule_weight_val = confidence[label_value] - (sum_confidence - confidence[label_value])

        if rule_weight_val <= reject_threshold:
            class_label.set_rejected() # TODO: check if it's modified globally or locally only
            return zero_weight

        return RuleWeightBasic(rule_weight_val)

    def get_training_set(self):
        return self._train_ds

    def __str__(self):
        return f"MoFGBML_Learning [defaultLimit={AbstractLearning._default_reject_threshold}]"

    def __copy__(self):
        return LearningBasic(self._train_ds)
