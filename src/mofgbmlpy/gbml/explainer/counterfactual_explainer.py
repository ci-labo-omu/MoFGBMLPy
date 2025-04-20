import copy
import numpy as np
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF


class CounterFactualExplainer:
    def __init__(
        self,
        fuzzy_rule,
        target_class,
        train_set,
        learner,
        confidence_loss_weight=0.5,
        area_computation_num_samples=100,
        learning_rate=0.05,
        max_num_epochs=30,
    ):
        self._fuzzy_rule = copy.deepcopy(fuzzy_rule)
        self._initial_class = fuzzy_rule.get_class_label()
        self._target_class = target_class
        self._train_set = train_set
        self._new_knowledge = copy.deepcopy(fuzzy_rule.get_knowledge())
        self._prev_knowledge = None
        self._learner = learner
        self._confidence_loss_weight = confidence_loss_weight
        self._area_computation_num_samples = area_computation_num_samples
        self._learning_rate = learning_rate
        self._max_num_epochs = max_num_epochs

    def compute_membership_values(self, fuzzy_sets, min_val=0, max_val=1):
        mfs = [fs.get_function() for fs in fuzzy_sets]

        x_samples = np.linspace(min_val, max_val, self._area_computation_num_samples)
        mf_values = np.zeros((len(fuzzy_sets), len(x_samples)), dtype=object)
        for i, mf in enumerate(mfs):
            mf_values[i] = [mf.get_value_py(x) for x in x_samples]

        return mf_values

    def compute_membership_area_data(self, mf_1_y, mf_2_y, step):
        union_value = np.zeros(mf_1_y.shape[0])
        intersection_value = np.zeros(mf_1_y.shape[0])
        mf_1_highest_area = np.zeros(mf_1_y.shape[0])
        mf_2_highest_area = np.zeros(mf_1_y.shape[0])

        for fs_i in range(mf_1_y.shape[0]):
            intersection_value[fs_i] = 0
            union_value[fs_i] = 0
            mf_1_highest_area[fs_i] = 0
            mf_2_highest_area[fs_i] = 0

            for i in range(mf_1_y.shape[1]):
                y_val = max(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                union_value[fs_i] += step * y_val

                y_val = min(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                intersection_value[fs_i] += step * y_val

                if mf_1_y[fs_i][i] > mf_2_y[fs_i][i]:
                    mf_1_highest_area[fs_i] += step
                else:
                    mf_2_highest_area[fs_i] += step

        return intersection_value, union_value, mf_1_highest_area, mf_2_highest_area

    def loss_functions(self, intersection_values, union_values):
        # Confidence loss
        # TODO: to be optimized, because for now all confidence are computed
        confidences = self._learner.calc_confidence_py(self._fuzzy_rule.get_antecedent(), self._train_set)

        confidence_initial_class = confidences[self._initial_class.get_class_label_value()]
        confidence_target_class = confidences[self._target_class.get_class_label_value()]
        confidence_loss = confidence_initial_class - confidence_target_class

        # Change loss
        change_loss = 0

        if intersection_values is not None and union_values is not None:
            for i in range(len(intersection_values)):
                change_loss += 1 - (intersection_values[i] / union_values[i])
                # print(f"Intersection: {intersection_values[i]}, Union: {union_values[i]}, Change Loss: {change_loss}")

            change_loss /= len(intersection_values)

        # Final loss
        return confidence_loss, change_loss

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
                if x >= mf_params[0] and x <= mf_params[1] and mf_params[0] != mf_params[1]
                else 0
            )
        elif param_index == 1:
            # dµ/db
            if x >= mf_params[0] and x <= mf_params[1] and mf_params[0] != mf_params[1]:
                return (mf_params[0] - x) / ((mf_params[1] - mf_params[0]) ** 2)
            elif x >= mf_params[1] and x <= mf_params[2] and mf_params[1] != mf_params[2]:
                return (mf_params[2] - x) / ((mf_params[2] - mf_params[1]) ** 2)
            else:
                return 0
        else:
            # dµ/dc
            return (
                (x - mf_params[1]) / ((mf_params[2] - mf_params[1]) ** 2)
                if x >= mf_params[1] and x <= mf_params[2] and mf_params[1] != mf_params[2]
                else 0
            )

    def _compute_gradient(
        self,
        antecedent_mf_value,
        fs_mf_values,
        mf_params,
        intersection_value,
        union_value,
        mf_1_highest_area,
        mf_2_highest_area,
    ):
        gradient = np.zeros((len(fs_mf_values[0]), 3), dtype=object)  # shape (num_fs, num_params)

        # dL_conf/d_membership_aq
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
            self._confidence_loss_weight
            * (
                num_p_initial_class * sum_all_mf_values
                - num_p * sum_initial_class_mf_values
                - num_p_target_class * sum_all_mf_values
                + num_p * sum_target_class_mf_values
            )
            / (sum_target_class_mf_values**2)
        )

        # dL_change/d_membership_aq
        if intersection_value is not None and union_value is not None:
            sum_values = 0
            for i in range(intersection_value.shape[0]):
                sum_values += mf_1_highest_area[i] * union_value[i] - intersection_value[i] * mf_2_highest_area[i]

            gradient[:, :] += (1 - self._confidence_loss_weight) * sum_values

        #######################################

        # (d_membership_aq/d_membership_aqi) * (d_membership_aqi/d_mf_params)
        for i in range(len(mf_params)):  # for all dimensions (i.e. fuzzy sets) in the antecedent
            if mf_params[i] is None or len(mf_params[i]) == 0:
                # Don't care FS
                gradient[i, :] = 0
                continue

            # sum
            summed_value = 0
            for j in range(num_p):  # fs_mf_values shape is num_patterns, num_dim
                if fs_mf_values[j, i] == 0:
                    continue
                summed_value += np.prod(fs_mf_values[j, :]) / fs_mf_values[j, i]

            summed_value /= num_p

            gradient[i, :] *= summed_value

            # print(f"Gradient {i}: {gradient[i, :]}")

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

    def train(self):
        # TODO: decouple fuzzy sets between vars (copy them in knowledge base and antecedent)

        antecedent = self._fuzzy_rule.get_antecedent()
        antecedent_indices = antecedent.get_antecedent_indices()

        fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
        for i, idx in enumerate(antecedent_indices):
            fuzzy_sets[i] = self._new_knowledge.get_fuzzy_set(i, idx)

        mf = [fs.get_function() for fs in fuzzy_sets]
        mf_params = [func.get_params() for func in mf]

        # self._fuzzy_rule.get_knowledge().plot_fuzzy_variables()

        prev_mf_values = None
        intersection_value, union_value, mf_1_highest_area, mf_2_highest_area = None, None, None, None
        for epoch in range(self._max_num_epochs):
            # forward
            fs_mf_values = np.array(
                [
                    self._fuzzy_rule.get_antecedent().get_membership_values(pattern.get_attributes_vector())
                    for pattern in self._train_set.get_patterns()
                ]
            )
            antecedent_mf_values = np.prod(fs_mf_values, axis=1)

            current_mf_values = self.compute_membership_values(fuzzy_sets, 0, 1)
            step = 1 / current_mf_values.shape[1]

            if prev_mf_values is not None:
                intersection_value, union_value, mf_1_highest_area, mf_2_highest_area = (
                    self.compute_membership_area_data(prev_mf_values, current_mf_values, step)
                )
            prev_mf_values = current_mf_values

            # loss
            conf_loss, change_loss = self.loss_functions(intersection_value, union_value)
            loss = self._confidence_loss_weight * conf_loss + (1 - self._confidence_loss_weight) * change_loss
            print(
                f"Epoch {epoch+1}/{self._max_num_epochs}, "
                f"Confidence Loss: {conf_loss},"
                f"Change Loss: {change_loss},"
                f"Total Loss: {loss}"
            )

            if conf_loss < 0:
                print("INFO: Early stopping: consequent changed")
                break

            # TODO: find a way to use batches ?
            # backward
            gradient = self._compute_gradient(
                antecedent_mf_values,
                fs_mf_values,
                mf_params,
                intersection_value,
                union_value,
                mf_1_highest_area,
                mf_2_highest_area,
            )
            # print(f"Gradient: {gradient}")

            # update params
            for fs_i, fs in enumerate(fuzzy_sets):
                if mf_params[fs_i] is None or len(mf_params[fs_i]) == 0:
                    continue  # Don't care FS

                new_params = np.zeros(len(mf_params[fs_i]), dtype=object)

                for p_i, param in enumerate(mf_params[fs_i]):
                    new_params[p_i] = mf_params[fs_i][p_i] - self._learning_rate * gradient[fs_i][p_i]
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

                # print(f"New Params: {new_params}")

                fs.set_function(TriangularMF(new_params[0], new_params[1], new_params[2]))

                mf_params[fs_i] = new_params
            self._fuzzy_rule.get_antecedent().set_knowledge(self._new_knowledge)

        self._fuzzy_rule.set_consequent(self._learner.learning(self._fuzzy_rule.get_antecedent(), self._train_set))
        self._fuzzy_rule.get_knowledge().plot_fuzzy_variables()

    def get_counterfactual(self):
        print(self._fuzzy_rule)
        self.train()
        # print(self._new_knowledge)

        self._fuzzy_rule.get_consequent()
        print(self._fuzzy_rule)

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
    # SIMPLE EXAMPLE

    # target_class = ClassLabelBasic(1)
    # antecedent_indices = np.array([1, 0, 2], dtype=int)
    #
    # knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    #
    # antecedent = Antecedent(antecedent_indices, knowledge)
    #
    # patterns = np.array(
    #     [
    #         Pattern(0, np.array([0.1, 0.2, 0.3]), ClassLabelBasic(0)),
    #         Pattern(1, np.array([0.4, 0.5, 0.6]), ClassLabelBasic(1)),
    #         Pattern(2, np.array([0.7, 0.8, 0.9]), ClassLabelBasic(0)),
    #     ]
    # )
    #
    # train_set = Dataset(size=3, n_dim=3, c_num=2, patterns=patterns)
    #
    # learner = LearningBasic(train_set)
    # consequent = learner.learning(antecedent, train_set)
    #
    # rule = RuleBasic(antecedent, consequent)
    #
    # explainer = CounterFactualExplainer(rule, target_class, train_set, learner, confidence_loss_weight=0.8)
    # explainer.get_counterfactual()

    # REAL EXAMPLE

    args = [
        "--data-name",
        "appendicitis",
        "--algorithm-id",
        "0",
        "--experiment-id",
        "0",
        "--train-file",
        "..\\..\\..\\..\\dataset\\appendicitis\\a0_0_appendicitis-10tra.dat",
        "--test-file",
        "..\\..\\..\\..\\dataset\\appendicitis\\a0_0_appendicitis-10tra.dat",
        "--terminate-evaluation",
        "1000",
        "--no-output-files",
        "--objectives",
        "error-rate",
        "num-rules",
    ]

    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    res = runner.main(args)

    non_dominated_solutions = res.X
    objectives_non_dominated_solutions = res.F

    sol1 = non_dominated_solutions[0]
    rule = sol1[0].get_var(0).get_rule()

    learner = LearningBasic(runner.get_train_set())
    explainer = CounterFactualExplainer(
        rule, ClassLabelBasic(1), runner.get_train_set(), learner, confidence_loss_weight=0.8, learning_rate=0.04
    )
    explainer.get_counterfactual()
