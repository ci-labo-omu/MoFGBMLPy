import os
import time
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf import TriangularMF
from pymoo.core.population import Population
from tqdm import tqdm

from mofgbmlpy.data.input import Input
from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.explainer.util import remove_duplicates
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain


class CounterFactualExplainerGradient:
    def __init__(
        self,
        classifier,
        changed_rule_index,
        target_class,
        test_set,
        confidence_loss_weight=0.9,
        area_computation_num_samples=100,
        learning_rate=4.0,
        max_num_epochs=100,
    ):
        self._problem = CounterfactualProblem(classifier, changed_rule_index, target_class, test_set=test_set)

        self._classifier_copy = copy.deepcopy(classifier)
        michigan_sol = classifier.get_var(changed_rule_index)
        self._factual_rule = copy.deepcopy(michigan_sol.get_rule())
        self._initial_class = self._factual_rule.get_class_label()
        self._target_class = target_class
        self._learner = michigan_sol.get_rule_builder().get_consequent_factory()
        self._train_set = self._learner.get_training_set()
        self._initial_knowledge = self._factual_rule.get_knowledge()
        self._confidence_loss_weight = confidence_loss_weight
        self._area_computation_num_samples = area_computation_num_samples
        self._learning_rate = learning_rate
        self._max_num_epochs = max_num_epochs

        antecedent = self._factual_rule.get_antecedent()
        antecedent_indices = antecedent.get_antecedent_indices()

        fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
        for i, idx in enumerate(antecedent_indices):
            fuzzy_sets[i] = self._initial_knowledge.get_fuzzy_set(i, idx)

        self._initial_mf_values = self.compute_membership_values(fuzzy_sets, 0, 1)

    def get_problem(self):
        return self._problem

    def compute_membership_values(self, fuzzy_sets, min_val=0, max_val=1):
        mfs = [fs.get_function() for fs in fuzzy_sets]

        x_samples = np.linspace(min_val, max_val, self._area_computation_num_samples)
        mf_values = np.zeros((len(fuzzy_sets), len(x_samples)), dtype=object)
        for i, mf in enumerate(mfs):
            mf_values[i] = [mf.get_value_py(x) for x in x_samples]

        return mf_values

    @staticmethod
    def compute_membership_area_data(mf_1_y, mf_2_y, step):
        union_value = np.zeros(mf_1_y.shape[0])
        intersection_value = np.zeros(mf_1_y.shape[0])
        mf_2_smallest_length = np.zeros(mf_1_y.shape[0])
        mf_2_highest_length = np.zeros(mf_1_y.shape[0])

        for fs_i in range(mf_1_y.shape[0]):
            intersection_value[fs_i] = 0
            union_value[fs_i] = 0
            mf_2_smallest_length[fs_i] = 0
            mf_2_highest_length[fs_i] = 0

            for i in range(mf_1_y.shape[1]):
                y_val = max(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                union_value[fs_i] += step * y_val

                y_val = min(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                intersection_value[fs_i] += step * y_val

                if mf_1_y[fs_i][i] > mf_2_y[fs_i][i]:
                    mf_2_smallest_length[fs_i] += step
                else:
                    mf_2_highest_length[fs_i] += step

        return intersection_value, union_value, mf_2_smallest_length, mf_2_highest_length

    def loss_functions(self, intersection_values, union_values, cf_rule):
        # Confidence loss
        # TODO: to be optimized, because for now all confidence are computed
        confidences = self._learner.calc_confidence_py(cf_rule.get_antecedent(), self._train_set)
        confidence_target_class = confidences[self._target_class.get_class_label_value()]
        confidence_loss = 1 - confidence_target_class

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
        mf_current_smallest_length,
        mf_current_highest_length,
    ):
        gradient = np.zeros((fs_mf_values.shape[1], 3), dtype=object)  # shape (num_fs, num_params)

        # dL_conf/d_membership_aq
        patterns_idx_initial_class, patterns_idx_target_class = CounterFactualExplainerGradient._filter_data_class(
            self._train_set, self._initial_class, self._target_class
        )

        patterns = self._train_set.get_patterns()
        num_p = len(patterns)

        sum_all_mf_values = np.sum(antecedent_mf_value)

        sum_target_class_mf_values = np.sum(antecedent_mf_value[patterns_idx_target_class])

        # # dL_change/d_membership_aq
        loss_change_derivative = 0
        if intersection_value is not None and union_value is not None:
            for i in range(intersection_value.shape[0]):
                loss_change_derivative += (
                    intersection_value[i] * mf_current_highest_length[i]
                    - mf_current_smallest_length[i] * union_value[i]
                ) / (union_value[i] ** 2)

        loss_change_derivative /= intersection_value.shape[0]

        for i in range(len(mf_params)):  # for all dimensions (i.e. fuzzy sets) in the antecedent
            if mf_params[i] is None or len(mf_params[i]) == 0:
                # Don't care FS
                gradient[i, :] = 0
                continue

            mean_value = np.zeros(len(mf_params[i]), dtype=object)
            for j in range(num_p):  # fs_mf_values shape is num_patterns, num_dim
                if fs_mf_values[j, i] == 0:
                    continue
                p = patterns[j]

                x = p.get_attributes_vector()

                # d_membership_aq / d_membership_aqi
                derivative1 = np.prod(fs_mf_values[j, :]) / fs_mf_values[j, i]

                if derivative1 == 0:
                    continue

                is_target_class = 1 if p.get_target_class() == self._target_class else 0

                confidence_loss_derivative = -(is_target_class * sum_all_mf_values - sum_target_class_mf_values) / (
                    sum_all_mf_values**2
                )
                # confidence_loss_derivative = is_target_class

                # d membership aqi / d mf params
                for k in range(len(mf_params[i])):
                    # TODO: put it into the mf function class directly maybe if needed
                    derivative2 = CounterFactualExplainerGradient.get_param_derivative(k, mf_params[i], x[i])
                    combined_loss_derivative = (
                        self._confidence_loss_weight * confidence_loss_derivative
                        + (1 - self._confidence_loss_weight) * loss_change_derivative
                    )

                    mean_value[k] += combined_loss_derivative * derivative1 * derivative2

            mean_value /= num_p
            gradient[i, :] = mean_value

            # print(f"Gradient {i}: {gradient[i, :]}")

        # for i in range(len(gradient)):
        #     print(f"Gradient {i}: {gradient[i, :]}")
        # print("#" * 50)
        return gradient

    def train(self, verbose=True):
        # TODO: decouple fuzzy sets between vars (copy them in knowledge base and antecedent)

        new_cf_rule = copy.deepcopy(self._factual_rule)
        new_knowledge = copy.deepcopy(self._initial_knowledge)

        antecedent = self._factual_rule.get_antecedent()
        antecedent_indices = antecedent.get_antecedent_indices()

        fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
        for i, idx in enumerate(antecedent_indices):
            fuzzy_sets[i] = new_knowledge.get_fuzzy_set(i, idx)

        mf = [fs.get_function() for fs in fuzzy_sets]
        mf_params = [func.get_params() for func in mf]

        # if verbose:
        #     new_cf_rule.plot_antecedent()

        losses = []
        steps_without_improvement = 0
        best_loss = float("inf")

        p_bar = tqdm(range(self._max_num_epochs), desc="Training...", disable=not verbose, unit=" epoch")
        for epoch in p_bar:
            # forward
            fs_mf_values = np.array(
                [
                    new_cf_rule.get_antecedent().get_membership_values(pattern.get_attributes_vector())
                    for pattern in self._train_set.get_patterns()
                ]
            )

            # print("FS MF Values:", np.sum(fs_mf_values, axis=1))

            antecedent_mf_values = np.prod(fs_mf_values, axis=1)

            current_mf_values = self.compute_membership_values(fuzzy_sets, 0, 1)
            step = 1 / current_mf_values.shape[1]

            intersection_value, union_value, mf_current_smallest_length, mf_current_highest_length = (
                CounterFactualExplainerGradient.compute_membership_area_data(
                    self._initial_mf_values, current_mf_values, step
                )
            )

            # loss
            conf_loss, change_loss = self.loss_functions(intersection_value, union_value, new_cf_rule)
            loss = self._confidence_loss_weight * conf_loss + (1 - self._confidence_loss_weight) * change_loss

            losses.append(loss)

            if verbose:
                p_bar.set_postfix(
                    confidence_loss=f"{conf_loss:.3f}",
                    change_loss=f"{change_loss:.3f}",
                    total_loss=f"{loss:.3f}",
                )

            # check if class is target
            if new_cf_rule.get_class_label() == self._target_class and not new_cf_rule.get_class_label().is_rejected():
                if verbose:
                    print("INFO: Early stopping: consequent changed")
                break

            # if no improvement in loss, stop training after 10 epochs
            elif abs(best_loss - losses[-1]) < 1e-6:
                steps_without_improvement += 1
                if steps_without_improvement >= 10:
                    if verbose:
                        print("INFO: Early stopping: no improvement in loss")
                    break
            else:
                steps_without_improvement = 0
                best_loss = losses[-1]

            # TODO: find a way to use batches ?
            # backward
            gradient = self._compute_gradient(
                antecedent_mf_values,
                fs_mf_values,
                mf_params,
                intersection_value,
                union_value,
                mf_current_smallest_length,
                mf_current_highest_length,
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

                new_params[0] = max(0, min(new_params[0], 1))
                new_params[1] = max(new_params[0], min(new_params[1], 1))
                new_params[2] = max(new_params[1], min(new_params[2], 1))

                fs.set_function(TriangularMF(new_params[0], new_params[1], new_params[2]))

                mf_params[fs_i] = new_params
            new_cf_rule.get_antecedent().set_knowledge(new_knowledge)

            new_cf_rule.set_consequent(self._learner.learning(new_cf_rule.get_antecedent(), self._train_set))

        # if verbose:
        # new_cf_rule.plot_antecedent()

        losses = np.array(losses)

        if verbose:
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.ylim(0, 1)
            plt.grid()
            plt.show()

        antecedent_indices = new_cf_rule.get_antecedent().get_antecedent_indices()
        fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
        for i, idx in enumerate(antecedent_indices):
            fuzzy_sets[i] = new_knowledge.get_fuzzy_set(i, idx)
        current_mf_values = self.compute_membership_values(fuzzy_sets, 0, 1)

        intersection_value, union_value, _, _ = CounterFactualExplainerGradient.compute_membership_area_data(
            self._initial_mf_values, current_mf_values, step=1 / current_mf_values.shape[1]
        )

        if new_cf_rule.get_class_label().is_rejected() or new_cf_rule.get_class_label() != self._target_class:
            if verbose:
                print(
                    f"Failure: Counterfactual rule class {new_cf_rule.get_class_label()} does not match target class {self._target_class} or is rejected."
                )
            return Population.new(X=np.array([], dtype=object), F=np.array([], dtype=float))

        michigan_solution = self._create_solution_object(new_cf_rule, new_knowledge)
        new_pop = Population.new(X=[[michigan_solution]])
        pop_F = self._problem.evaluate(new_pop.get("X"), return_values_of=["F"])
        new_pop.set("F", pop_F)

        return new_pop

    def _create_solution_object(self, new_cf_rule, new_knowledge):
        michigan_sol = copy.deepcopy(self._classifier_copy.get_var(self._problem.get_changed_rule_index()))
        michigan_sol.resize_objectives(self._problem.n_obj)
        michigan_sol.set_vars(new_cf_rule.get_antecedent().get_antecedent_indices())
        michigan_sol.set_knowledge(new_knowledge)

        return michigan_sol
