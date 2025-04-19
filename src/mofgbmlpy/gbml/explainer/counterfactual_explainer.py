import copy
import numpy as np
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.data.dataset import Dataset
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)


class CounterFactualExplainer:
    def __init__(self, fuzzy_rule, target_class, train_set, learner):
        self._fuzzy_rule = copy.deepcopy(fuzzy_rule)
        self._initial_class = fuzzy_rule.get_class_label()
        self._target_class = target_class
        self._train_set = train_set
        self._new_knowledge = copy.deepcopy(fuzzy_rule.get_knowledge())
        self._learner = learner

    def loss_function(self):
        # TODO: to be optimized, because for now all confidence are computed
        confidences = self._learner.calc_confidence_py(self._fuzzy_rule.get_antecedent(), self._train_set)

        confidence_initial_class = confidences[self._initial_class.get_class_label_value()]
        confidence_target_class = confidences[self._target_class.get_class_label_value()]
        return confidence_initial_class - confidence_target_class

    @staticmethod
    def _filter_data_class(dataset, searched_class1, searched_class2):
        patterns = dataset.get_patterns()

        filtered_data1_idx = []
        filtered_data2_idx = []

        for i, x in enumerate(patterns):
            if x.get_target_class() == searched_class1:
                filtered_data1_idx.append(i)
            elif x.get_target_class() == searched_class2:
                filtered_data2_idx.append(i)
        return filtered_data1_idx, filtered_data2_idx

    @staticmethod
    def get_param_derivative(param_index, mf_params, x):
        if param_index == 0:
            # dµ/da
            return (
                (x - mf_params[1]) / ((mf_params[1] - mf_params[0]) ** 2)
                if x >= mf_params[0] and x <= mf_params[1]
                else 0
            )
        elif param_index == 1:
            # dµ/db
            if x >= mf_params[0] and x <= mf_params[1]:
                return (mf_params[0] - x) / ((mf_params[1] - mf_params[0]) ** 2)
            elif x >= mf_params[1] and x <= mf_params[2]:
                return (mf_params[2] - x) / ((mf_params[2] - mf_params[1]) ** 2)
            else:
                return 0
        else:
            # dµ/dc
            return (
                (x - mf_params[1]) / ((mf_params[2] - mf_params[1]) ** 2)
                if x >= mf_params[1] and x <= mf_params[2]
                else 0
            )

    def _compute_gradient(self, antecedent_mf_value, fs_mf_values, mf_params):
        gradient = np.zeros((len(fs_mf_values[0]), 3), dtype=object)  # shape (num_fs, num_params)

        # dL/d_membership_aq
        patterns_idx_initial_class, patterns_idx_target_class = self._filter_data_class(
            self._train_set, self._initial_class, self._target_class
        )

        num_p_initial_class = len(patterns_idx_initial_class)
        num_p_target_class = len(patterns_idx_target_class)

        patterns = self._train_set.get_patterns()
        num_p = len(patterns)

        sum_all_mf_values = np.sum(antecedent_mf_value)

        sum_initial_class_mf_values = np.sum(antecedent_mf_value[patterns_idx_initial_class])
        sum_target_class_mf_values = np.sum(antecedent_mf_value[patterns_idx_target_class])

        gradient[:, :] = (
            num_p_initial_class * sum_all_mf_values
            - num_p * sum_initial_class_mf_values
            - num_p_target_class * sum_all_mf_values
            + num_p * sum_target_class_mf_values
        ) / (sum_target_class_mf_values**2)

        #######################################

        # (d_membership_aq/d_membership_aqi) * (d_membership_aqi/d_mf_params)
        prod_all_fs_mf_values = np.prod(fs_mf_values)
        for i in range(len(mf_params)):  # for all dimensions (i.e. fuzzy sets) in the antecedent
            if mf_params[i] is None or len(mf_params[i]) == 0:
                # Don't care FS
                gradient[i, :] = 0
                continue

            # d membership aq / d membership aqi
            gradient[i, :] *= prod_all_fs_mf_values / fs_mf_values[i]

            # d membership aqi / d mf params
            # TODO: put it into the mf function class directly (here temporarily for testing)
            for j in range(len(mf_params[i])):
                mean_value = 0
                for p in patterns:
                    x = p.get_attributes_vector()
                    mean_value += self.get_param_derivative(j, mf_params[i], x[i])
                mean_value /= num_p
                gradient[i, j] *= mean_value

        return gradient

    def train(self, num_epochs=10, learning_rate=0.2):
        antecedent = self._fuzzy_rule.get_antecedent()
        antecedent_indices = antecedent.get_antecedent_indices()

        fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
        for i, idx in enumerate(antecedent_indices):
            fuzzy_sets[i] = self._new_knowledge.get_fuzzy_set(i, idx)

        mf = [fs.get_function() for fs in fuzzy_sets]
        mf_params = [func.get_params() for func in mf]

        # self._fuzzy_rule.get_knowledge().plot_fuzzy_variables()

        for epoch in range(num_epochs):
            # forward
            fs_mf_values = [
                self._fuzzy_rule.get_antecedent().get_membership_values(pattern.get_attributes_vector())
                for pattern in self._train_set.get_patterns()
            ]
            antecedent_mf_values = np.prod(fs_mf_values, axis=1)

            # loss
            # TODO: change loss function, because it doesn't consider the smallest change here
            loss = self.loss_function()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

            # TODO: find a way to use batches ?
            # backward
            gradient = self._compute_gradient(antecedent_mf_values, fs_mf_values, mf_params)

            # update params
            for fs_i, fs in enumerate(fuzzy_sets):
                if mf_params[fs_i] is None or len(mf_params[fs_i]) == 0:
                    continue  # Don't care FS

                new_params = np.zeros(len(mf_params[fs_i]), dtype=object)

                for p_i, param in enumerate(mf_params[fs_i]):
                    new_params[p_i] = mf_params[fs_i][p_i] - learning_rate * gradient[fs_i][p_i]
                    if new_params[p_i] < 0:
                        new_params[p_i] = 0
                    elif new_params[p_i] > 1:
                        new_params[p_i] = 1
                    elif np.isnan(new_params[p_i]):
                        new_params[p_i] = mf_params[fs_i][p_i]

                # fix a <= b <= c

                for p_i, param in enumerate(new_params):
                    prev_val = new_params[p_i - 1] if p_i > 0 else 0
                    next_val = new_params[p_i + 1] if p_i < len(new_params) - 1 else 1

                    # repair
                    if new_params[p_i] < prev_val:
                        new_params[p_i] = prev_val
                    elif new_params[p_i] > next_val:
                        new_params[p_i] = next_val

                    fs.get_function().set_param_value(p_i, new_params[p_i])

                mf_params[fs_i] = new_params
            self._fuzzy_rule.get_antecedent().set_knowledge(self._new_knowledge)

        # print(f"next ({antecedent_indices[0]})", self._new_knowledge.get_fuzzy_set(0, antecedent_indices[0]).get_function().get_params())
        self._fuzzy_rule.get_knowledge().plot_fuzzy_variables()

    def get_counterfactual(self):
        self.train()
        print(self._new_knowledge)
        #
        # new_knowledge, new_classifier = ...
        #
        # knowledge_copy = copy.deepcopy(fuzzy_classifier.get_var(0).get_rule().get_knowledge())
        #
        # for var in solution_copy.get_vars():
        #     var.get_rule().get_antecedent().set_knowledge(new_knowledge)
        #
        # return new_classifier


if __name__ == "__main__":
    # Test the CounterFactualExplainer class
    target_class = ClassLabelBasic(1)
    antecedent_indices = np.array([1, 0, 2], dtype=int)

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()

    antecedent = Antecedent(antecedent_indices, knowledge)

    patterns = np.array(
        [
            Pattern(0, np.array([0.1, 0.2, 0.3]), ClassLabelBasic(0)),
            Pattern(1, np.array([0.4, 0.5, 0.6]), ClassLabelBasic(1)),
            Pattern(2, np.array([0.7, 0.8, 0.9]), ClassLabelBasic(0)),
        ]
    )

    train_set = Dataset(size=3, n_dim=3, c_num=2, patterns=patterns)

    learner = LearningBasic(train_set)
    consequent = learner.learning(antecedent, train_set)

    rule = RuleBasic(antecedent, consequent)

    explainer = CounterFactualExplainer(rule, target_class, train_set, learner)
    explainer.get_counterfactual()
