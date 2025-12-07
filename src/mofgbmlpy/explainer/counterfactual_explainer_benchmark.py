import copy
import random
import time
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.optimize import minimize
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from mofgbmlpy.explainer.counterfactual_explainer_gradient import CounterFactualExplainerGradient
from mofgbmlpy.explainer.gbml.crowding_function_x import CrowdingFunctionX
from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.explainer.gbml.fuzzy_sets_sampling import FuzzySetsSampling
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_mutation import FuzzySetsMutation
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_crossover import FuzzySetsCrossover
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
import os
import numpy as np
from mofgbmlpy.explainer.gbml.fuzzy_sets_eliminate_duplicates import FuzzySetsEliminateDuplicates
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_survival import FuzzySetsSurvival
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain
import pandas as pd

from mofgbmlpy.explainer.util import append_rule_classifier


class CounterFactualExplainerBenchmark:
    @staticmethod
    def compute_diversity(pop):
        if pop is None or len(pop) <= 1:
            return 0.0

        distances = CrowdingFunctionX.calc_crowding_distance(pop.get("X"))
        return np.mean(distances)

    @staticmethod
    def __append_metric(results, metric_name, value, extend=False):
        if metric_name not in results:
            results[metric_name] = []
        if extend:
            results[metric_name].extend(value)
        else:
            results[metric_name].append(value)
        return results

    @staticmethod
    def all_metrics_eval(explainer, solutions):
        results = {}
        problem = explainer.get_problem()
        initial_train_error_rate = problem.error_rate(use_test_set=False, initial_classifier=True)
        initial_test_error_rate = problem.error_rate(use_test_set=True, initial_classifier=True)
        initial_num_wins_list, initial_num_successes_list = problem.num_wins_and_successes(initial_classifier=True)
        rule_index = problem.get_changed_rule_index()
        initial_num_wins = initial_num_wins_list[rule_index]
        initial_num_successes = initial_num_successes_list[rule_index]
        initial_num_wins_freq = initial_num_wins / np.sum(initial_num_wins_list)
        initial_num_successes_freq = initial_num_successes / np.sum(initial_num_successes_list)

        for sol in solutions:
            results = CounterFactualExplainerBenchmark.__append_metric(
                results, "confidence_loss", problem.conf_loss(sol)
            )
            results = CounterFactualExplainerBenchmark.__append_metric(results, "change_loss", problem.change_loss(sol))
            results = CounterFactualExplainerBenchmark.__append_metric(
                results, "num_changed_features", problem.num_changed_features_loss(sol)
            )

            # for replace in [True, False]:
            replace = False
            for use_test in [True, False]:
                metric_name = f"{'test' if use_test else 'train'}_error_rate_{'replace' if replace else 'append'}"
                error_rate_val = problem.error_rate(sol, replace=replace, use_test_set=use_test)
                results = CounterFactualExplainerBenchmark.__append_metric(results, metric_name, error_rate_val)
                init_val = initial_test_error_rate if use_test else initial_train_error_rate
                var_value = error_rate_val - init_val
                results = CounterFactualExplainerBenchmark.__append_metric(
                    results, f"{metric_name}_initial_variation", var_value
                )

            # for repl, txt in [(False, "append"), (True, "replace")]:
            repl, txt = False, "append"
            num_wins_list, num_successes_list = problem.num_wins_and_successes(sol, replace=replace)
            idx = rule_index if repl else len(num_wins_list) - 1
            for freq in [False, True]:
                val_num_wins, val_num_successes = num_wins_list[idx], num_successes_list[idx]

                if freq:
                    val_num_wins /= np.sum(num_wins_list) if np.sum(num_wins_list) > 0 else 0
                    val_num_successes /= np.sum(num_successes_list) if np.sum(num_successes_list) > 0 else 0
                    val_initial_num_wins, val_initial_num_successes = (
                        initial_num_wins_freq,
                        initial_num_successes_freq,
                    )
                    txt += "_freq"
                else:
                    val_initial_num_wins, val_initial_num_successes = initial_num_wins, initial_num_successes

                results = CounterFactualExplainerBenchmark.__append_metric(results, f"num_wins_{txt}", val_num_wins)
                results = CounterFactualExplainerBenchmark.__append_metric(
                    results, f"num_successes_{txt}", val_num_successes
                )
                results = CounterFactualExplainerBenchmark.__append_metric(
                    results, f"num_wins_initial_variation_{txt}", val_num_wins - val_initial_num_wins
                )
                results = CounterFactualExplainerBenchmark.__append_metric(
                    results, f"num_successes_initial_variation_{txt}", val_num_successes - val_initial_num_successes
                )

        return results

    @staticmethod
    def get_stats(metric_vals):
        return {
            "min": np.min(metric_vals),
            "max": np.max(metric_vals),
            "mean": np.mean(metric_vals),
            "std": np.std(metric_vals),
            "median": np.median(metric_vals),
            "q1": np.percentile(metric_vals, 25),
            "q3": np.percentile(metric_vals, 75),
        }

    @staticmethod
    def main_benchmark(
        explainer_class,
        num_classes,
        classifiers,
        out_path,
        seed=2017,
        test_dataset=None,
        data_name="",
        test_name="",
        **kwargs,
    ):
        if os.path.exists(out_path):
            print(f"Output path {out_path} already exists (skipped).")
            return

        if "objectives" in kwargs and explainer_class == CounterFactualExplainerGradient:
            del kwargs["objectives"]

        np.random.seed(seed)
        random.seed(seed)

        metrics_values = {"time_in_seconds": [], "num_sols": [], "diversity": []}
        metrics_stats = {}
        num_failures = 0

        class_labels = [ClassLabelBasic(c) for c in range(num_classes)]

        desc = f"Running test {test_name}"
        if data_name != "":
            desc += f" on {data_name}"

        num_runs = CounterFactualExplainerBenchmark.get_num_iters(classifiers, num_classes)

        # best_train_error_rate_val = float("inf")
        # best_train_error_rate_sol_data = None

        with tqdm(total=num_runs, desc=desc) as pbar:
            # for i_classifier, p_sol in enumerate(classifiers):
            for p_sol in classifiers:
                num_rules = p_sol[0].get_num_vars()
                for i_var in range(num_rules):
                    class_label = p_sol[0].get_var(i_var).get_class_label()
                    for target_class in class_labels:
                        if target_class == class_label:
                            continue
                        pbar.update(1)
                        start = time.time()

                        explainer = explainer_class(p_sol[0], i_var, target_class, test_dataset, **kwargs)
                        solutions = explainer.train(verbose=False)

                        if solutions is None or len(solutions) == 0:
                            num_failures += 1
                            continue

                        end = time.time()

                        # print("new", solutions[0])
                        # print("old", p_sol[0].get_var(i_var))

                        current_metrics_vals = CounterFactualExplainerBenchmark.all_metrics_eval(
                            explainer, solutions.get("X").flatten()
                        )

                        metrics_values["time_in_seconds"].append(end - start)
                        metrics_values["num_sols"].append(len(solutions))
                        metrics_values["diversity"].append(
                            CounterFactualExplainerBenchmark.compute_diversity(solutions)
                        )

                        # for train_error_rate_idx, train_error_rate_val in enumerate(
                        #     current_metrics_vals["train_error_rate_append_initial_variation"]
                        # ):
                        #     if train_error_rate_val < best_train_error_rate_val:
                        #         best_train_error_rate_val = train_error_rate_val
                        #         best_train_error_rate_sol_data = {
                        #             "classifier": i_classifier,
                        #             "changed_rule_index": i_var,
                        #             "target_class": target_class,
                        #             "sol_idx": train_error_rate_idx,
                        #         }

                        for name, vals in current_metrics_vals.items():
                            if name not in metrics_values:
                                metrics_values[name] = []
                                metrics_stats[name] = {}
                            metrics_values[name].extend(vals)

                            current_stats = CounterFactualExplainerBenchmark.get_stats(vals)
                            for stat_name, val in current_stats.items():
                                if stat_name not in metrics_stats[name]:
                                    metrics_stats[name][stat_name] = []
                                metrics_stats[name][stat_name].append(val)

        # print(f"Best train error rate variation found: {best_train_error_rate_val:.3f} for solution: {best_train_error_rate_sol_data}")

        dataframe_data = {
            "time": metrics_values["time_in_seconds"],
            "num_sols": metrics_values["num_sols"],
            "diversity": metrics_values["diversity"],
        }

        for metric_name, stats_dict in metrics_stats.items():
            for stat_name, val in stats_dict.items():
                dataframe_data[f"{metric_name}_{stat_name}"] = val

        df = pd.DataFrame(dataframe_data)

        os.makedirs(out_path, exist_ok=False)
        df.to_csv(f"{out_path}\\results.csv", index=False)

        with open(f"{out_path}\\results_summary.txt", "w") as f:
            f.write(f"Number of runs: {num_runs}\n")
            f.write(f"Number of failures: {num_failures}\n")

            if num_runs - num_failures > 0:
                for metric_name, vals in metrics_values.items():
                    stats = CounterFactualExplainerBenchmark.get_stats(vals)
                    for stat_name, val in stats.items():
                        f.write(f"{metric_name}_{stat_name}: {val:.3f}\n")
                    f.write("\n")

    @staticmethod
    def main_plot_single(
        explainer_class, classifiers, cl_idx=0, r_idx=0, c_target=0, sol_idx=None, plot=True, test_dataset=None, var_names=None, class_labels=None, **kwargs
    ):
        classifier = classifiers[cl_idx][0]

        target_class = ClassLabelBasic(c_target)
        if classifier.get_var(r_idx).get_class_label() == target_class:
            target_class = ClassLabelBasic(min(0, 1 - c_target))

        if "objectives" in kwargs and explainer_class == CounterFactualExplainerGradient:
            del kwargs["objectives"]

        explainer = explainer_class(classifier, r_idx, target_class, test_set=test_dataset, **kwargs)
        non_dominated_solutions = explainer.train(verbose=True)


        if non_dominated_solutions is None or len(non_dominated_solutions) == 0:
            print("No solutions found")
            return

        if len(non_dominated_solutions) > 0 and sol_idx is not None and sol_idx < len(non_dominated_solutions):
            sol = non_dominated_solutions[sol_idx]
            non_dominated_solutions = Population.new(X=np.array([sol.X]), F=np.array([sol.F]))

        rules = non_dominated_solutions.get("X").flatten()
        #
        # problem = explainer.get_problem()
        # initial_error_rate = problem.error_rate(initial_classifier=True)
        # not_worse_indices = []
        # for i, sol in enumerate(non_dominated_solutions):
        #     train_err_var = problem.error_rate(sol.X[0], replace=False) - initial_error_rate
        #     print(f"Solution {i} Train error rate variation: {train_err_var:.4f}")
        #     if train_err_var <= 0:
        #         not_worse_indices.append(i)
        # remove not in not_worse_indices
        # rules = [rules[i] for i in not_worse_indices]

        # plot the results
        if plot:
            plot = Scatter(title="NSGA-II")
            plot.add(non_dominated_solutions.get("F"))
            plot.axis_labels = explainer.get_problem().get_objective_names()
            _ = plot.show()

        print("Factual rule:")
        factual_rule = classifier.get_var(r_idx)
        print(factual_rule)
        if plot:
            factual_rule.get_rule().plot_antecedent("Factual rule", var_names)

        print("Rules of non-dominated solutions:")

        for i in range(len(rules)):
            rule = rules[i]
            print(rule)
            if i < 10 and plot:
                rule.get_rule().plot_antecedent(f"CF Rule {i+1}", var_names)

        if plot and len(rules) > 10:
            print("Some rule plots were not displayed because there are too many rules")

        # current_metrics_vals = CounterFactualExplainerBenchmark.all_metrics_eval(
        #     explainer, non_dominated_solutions.get("X").flatten()
        # )
        # print("Metrics values for non-dominated solutions:")
        # for name, vals in current_metrics_vals.items():
        #     stats = CounterFactualExplainerBenchmark.get_stats(vals)
        #     print(f"{name}: {stats}")

        if plot and len(rules) < 3:
            train_set = explainer.get_problem().get_train_set()
            CounterFactualExplainerBenchmark.compare_classifiers(classifier, rules, train_set, var_names, class_labels)

        return rules

    @staticmethod
    def compare_classifiers(initial_classifier, cf_rules, train_set, var_names=None, class_labels=None):
        X, y = train_set.get_scikit_xy()

        # Compare the two classifiers
        initial_classifier_sk = initial_classifier.create_scikit_classifier()

        initial_classifier_sk.fit(X, y)

        # Decision boundary plot
        initial_classifier_sk.plot_decision_boundaries(
            X, y, title="Initial Classifier Decision Boundaries", fixed_vals=[0.5, None, None, 0.5], var_names=var_names, class_labels=class_labels
        )

        # Confusion matrix plot
        initial_classifier_sk.plot_conf_matrix(X, y, title="Initial Classifier Confusion Matrix", class_labels=class_labels)

        for cf_rule in cf_rules:
            new_cl = append_rule_classifier(initial_classifier, cf_rule, train_set=train_set)
            new_cl = new_cl.create_scikit_classifier()
            new_cl.fit(X, y)

            # Decision boundary plot
            new_cl.plot_decision_boundaries(
                X, y, title="New Classifier Decision Boundaries", fixed_vals=[0.5, None, None, 0.5], var_names=var_names, class_labels=class_labels
            )

            # Confusion matrix plot
            new_cl.plot_conf_matrix(X, y, title="New Classifier Confusion Matrix", class_labels=class_labels)

    @staticmethod
    def param_search(
        explainer_class,
        num_classes,
        classifiers,
        out_path,
        param_name,
        num_experiments=11,
        min_val=0.0,
        max_val=1.0,
        test_dataset=None,
        data_name="",
        is_int=False,
        vals=None,
        **kwargs,
    ):
        if vals is not None:
            param_vals = np.array(vals, dtype=int if is_int else float)
        else:
            if is_int:
                param_vals = np.linspace(min_val, max_val, num_experiments, dtype=int)
            else:
                param_vals = np.linspace(min_val, max_val, num_experiments)

        if vals is not None:
            step = min([abs(param_vals[i + 1] - param_vals[i]) for i in range(len(param_vals) - 1)])
        else:
            step = param_vals[1] - param_vals[0]
        precision = max(0, -int(np.floor(np.log10(step))))

        for param_val in param_vals:
            param_val_str = f"{param_val:.{precision}f}" if not is_int else f"{param_val}"
            current_path = os.path.join(out_path, f"{param_name}_{param_val_str}")
            new_kwargs = kwargs.copy()
            new_kwargs[param_name] = param_val

            # try:
            if os.path.exists(current_path):
                print(f"Path {current_path} already exists (skipped).")
                continue
            test_name = f"Param search {param_name}={param_val_str}"
            CounterFactualExplainerBenchmark.main_benchmark(
                explainer_class,
                num_classes,
                classifiers,
                out_path=current_path,
                test_dataset=test_dataset,
                data_name=data_name,
                test_name=test_name,
                **new_kwargs,
            )
            # except Exception as e:
            #     print(f"Error processing {param_name}={param_val:.2f}: {e}")

    @staticmethod
    def get_num_iters(classifiers, num_classes):
        num_rules = 0
        num_classes_checks = num_classes - 1

        for p_sol in classifiers:
            num_rules += p_sol[0].get_num_vars()

        return num_rules * num_classes_checks
