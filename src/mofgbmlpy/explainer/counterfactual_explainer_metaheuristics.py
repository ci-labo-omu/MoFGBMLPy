import random
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.optimize import minimize
from tqdm import tqdm

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


class CounterFactualExplainerMetaheuristics:
    def __init__(self, classifier, changed_rule_index, target_class, test_set, mutation_fs_type_prob=0.5, sampling_noise_str=0.1, mutation_prob=0.7, mutated_param_prob=0.6, crossover_prob=0.7, crossover_p1_selected_prob=0.5, sampling_fs_type_prob=0.0, use_search_space_crowding=False, objectives=["confidence_loss", "change_loss"]):
        self._problem = CounterfactualProblem(classifier, changed_rule_index, target_class, test_set=test_set, objectives=objectives)

        self._sampling = FuzzySetsSampling(sampling_noise_str, sampling_fs_type_prob)
        self._mutation = FuzzySetsMutation(mutation_prob, mutated_param_prob, mutation_fs_type_prob)
        self._crossover = FuzzySetsCrossover(crossover_prob, crossover_p1_selected_prob)
        self._eliminate_duplicates = FuzzySetsEliminateDuplicates(self._problem)
        self._survival = FuzzySetsSurvival(use_search_space_crowding=use_search_space_crowding)

    def get_target_class(self):
        return self._problem.get_target_class()

    @staticmethod
    def _save_generations_video_pymoo(history, out_path, file_name_without_extension):
        """Save the generations of a pymoo optimization history as a video.

        Args:
            history (list): List of pymoo optimization history objects.
            out_path (str): Path to save the video to.
            title (str): Title of the video.
        """
        os.makedirs(out_path, exist_ok=True)
        out_file_path = os.path.join(out_path, file_name_without_extension + ".mp4")

        with Recorder(Video(out_file_path)) as rec:
            for entry in history:
                sc = Scatter(title=("Gen %s" % entry.n_gen))

                full_pop = entry.pop.get("F")
                opt_pop = entry.opt.get("F")
                full_pop = np.array([x for x in full_pop if x not in opt_pop])

                if len(full_pop) != 0:
                    sc.add(full_pop, color="blue")
                elif len(opt_pop) == 0:
                    raise ValueError("No data to plot")
                sc.add(opt_pop, color="red")

                sc.do()

                rec.record()

    @staticmethod
    def remove_non_target_class_solutions(solutions, target_class):
        """Remove solutions that do not belong to the target class.

        Args:
            solutions (Population): List of solutions to filter.
            target_class (ClassLabelBasic): The target class to keep.

        Returns:
            list: Filtered list of solutions that belong to the target class.
        """
        filtered_solutions_X = []
        filtered_solutions_F = []

        if solutions is None:
            return None

        for i in range(len(solutions)):
            solution = solutions[i]
            rule = solution.X[0]
            if rule.get_class_label() == target_class and not rule.get_class_label().is_rejected():
                filtered_solutions_X.append(solution.X)
                filtered_solutions_F.append(solution.F)

        filtered_solutions_X = np.array(filtered_solutions_X, dtype=object)
        filtered_solutions_F = np.array(filtered_solutions_F, dtype=float)

        return Population.new(X=filtered_solutions_X, F=filtered_solutions_F)

    def train(self, n_gen=100, pop_size=60, verbose=True):
        # self._problem.get_fuzzy_rule().plot_antecedent()

        termination = get_termination("n_gen", n_gen)

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=self._sampling,
            crossover=self._crossover,
            mutation=self._mutation,  # should consider bounds and conditions of membership functions params
            eliminate_duplicates=False,  # self._eliminate_duplicates,
            save_history=False,  # True,
            survival=self._survival,
        )

        res = minimize(self._problem, algorithm, seed=41, verbose=verbose, termination=termination)

        non_dominated_solutions = res.opt

        if non_dominated_solutions is None:
            return Population.new(X=np.array([], dtype=object), F=np.array([], dtype=float)), np.array([], dtype=object)

        non_dominated_solutions = self._eliminate_duplicates.do(non_dominated_solutions)

        if len(non_dominated_solutions) == 0:
            return Population.new(X=np.array([], dtype=object), F=np.array([], dtype=float)), np.array([], dtype=object)

        non_dominated_solutions = self.remove_non_target_class_solutions(non_dominated_solutions, self.get_target_class())

        return non_dominated_solutions

    def compute_diversity(self, pop):
        # for now we simply use the average distance between all pairs of solutions
        if pop is None or len(pop) == 0:
            return 0.0
        num_sols = len(pop)
        distance = self._eliminate_duplicates.calc_dist(pop)
        if num_sols <= 1:
            return 0.0

        sum_dist = distance.sum() / 2
        return sum_dist / (num_sols * (num_sols - 1) // 2)

    @staticmethod
    def __append_metric(results, metric_name, value, extend=False):
        if metric_name not in results:
            results[metric_name] = []
        if extend:
            results[metric_name].extend(value)
        else:
            results[metric_name].append(value)
        return results

    def all_metrics_eval(self, solutions):
        results = {}
        initial_train_error_rate = self._problem.error_rate(use_test_set=False, initial_classifier=True)
        initial_test_error_rate = self._problem.error_rate(use_test_set=True, initial_classifier=True)
        initial_num_wins, initial_num_successes = self._problem.num_wins_and_successes(initial_classifier=True)

        for sol in solutions:
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "confidence_loss", self._problem.conf_loss(sol))
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "change_loss", self._problem.change_loss(sol))
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_changed_features", self._problem.num_changed_features_loss(sol))
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "train_error_rate", self._problem.error_rate(sol, use_test_set=False))
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "test_error_rate", self._problem.error_rate(sol, use_test_set=True))
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "train_error_rate_variation_replace", self._problem.error_rate(sol, replace=True, use_test_set=False) - initial_train_error_rate)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "train_error_rate_variation_append", self._problem.error_rate(sol, replace=False, use_test_set=False) - initial_train_error_rate)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "test_error_rate_variation_replace", self._problem.error_rate(sol, replace=True, use_test_set=True) - initial_test_error_rate)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "test_error_rate_variation_append", self._problem.error_rate(sol, replace=False, use_test_set=True) - initial_test_error_rate)

            num_wins, num_successes = self._problem.num_wins_and_successes(sol, replace=True)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_wins_replace", num_wins, extend=True)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_successes_replace", num_successes, extend=True)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_wins_variation_replace", num_wins - initial_num_wins, extend=True)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_successes_variation_replace", num_successes - initial_num_successes, extend=True)

            num_wins, num_successes = self._problem.num_wins_and_successes(sol, replace=False)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_wins_append", num_wins, extend=True)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_successes_append", num_successes, extend=True)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_wins_variation_append", num_wins - np.append(initial_num_wins, 0), extend=True)
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_successes_variation_append", num_successes - np.append(initial_num_successes, 0), extend=True)

        return results


def get_stats(metric_vals):
    return {
        "min": np.min(metric_vals),
        "max": np.max(metric_vals),
        "mean": np.mean(metric_vals),
        "std": np.std(metric_vals),
    }


def main_benchmark(num_classes, classifiers, out_path, seed=2017, test_dataset=None, **kwargs):
    if os.path.exists(out_path):
        print(f"Output path {out_path} already exists (skipped).")
        return

    np.random.seed(seed)
    random.seed(seed)

    metrics_values = {"time_in_seconds": [], "num_sols": [], "diversity": []}
    metrics_stats = {}
    num_failures = 0
    num_runs = 0

    class_labels = [ClassLabelBasic(c) for c in range(num_classes)]

    for p_sol in tqdm(classifiers):
        num_rules = p_sol[0].get_num_vars()
        for i_var in range(num_rules):
            class_label = p_sol[0].get_var(i_var).get_class_label()
            for target_class in class_labels:
                if target_class == class_label:
                    continue
                start = time.time()

                try:
                    explainer = CounterFactualExplainerMetaheuristics(p_sol[0], i_var, target_class, test_dataset, **kwargs)
                    solutions = explainer.train(n_gen=60, pop_size=60, verbose=False)

                    end = time.time()

                    metrics_values["time_in_seconds"].append(end - start)
                    metrics_values["num_sols"].append(len(solutions))
                    metrics_values["diversity"].append(explainer.compute_diversity(solutions))

                    current_metrics_vals = explainer.all_metrics_eval(solutions.get("X").flatten())

                    for (name, vals) in current_metrics_vals.items():
                        if name not in metrics_values:
                            metrics_values[name] = []
                            metrics_stats[name] = {}
                        metrics_values[name].extend(vals)

                        current_stats = get_stats(vals)
                        for (stat_name, val) in current_stats.items():
                            if stat_name not in metrics_stats[name]:
                                metrics_stats[name][stat_name] = []
                            metrics_stats[name][stat_name].append(val)

                except Exception as e:
                    # print(f"Failure: {e}")
                    raise e
                    num_failures += 1
                num_runs += 1

    dataframe_data = {
        "time": metrics_values["time_in_seconds"],
        "num_sols": metrics_values["num_sols"],
        "diversity": metrics_values["diversity"],
    }

    for (metric_name, stats_dict) in metrics_stats.items():
        for stat_name, val in stats_dict.items():
            dataframe_data[f"{metric_name}_{stat_name}"] = val

    df = pd.DataFrame(dataframe_data)

    os.makedirs(out_path, exist_ok=False)
    df.to_csv(f"{out_path}\\results.csv", index=False)

    with open(f"{out_path}\\results_summary.txt", "w") as f:
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Number of failures: {num_failures}\n")

        if num_runs - num_failures > 0:
            for (metric_name, vals) in metrics_values.items():
                stats = get_stats(vals)
                for stat_name, val in stats.items():
                    f.write(f"{metric_name}_{stat_name}: {val:.3f}\n")
                f.write("\n")


def main_plot_single(classifiers, test_dataset=None, **kwargs):
    classifier = classifiers[0][0]
    changed_rule_index = 0

    print(classifier)

    target_class = ClassLabelBasic(0)
    if classifier.get_var(changed_rule_index).get_class_label() == target_class:
        target_class = ClassLabelBasic(1)
    explainer = CounterFactualExplainerMetaheuristics(classifier, changed_rule_index, target_class, test_set=test_dataset, **kwargs)
    non_dominated_solutions = explainer.train(n_gen=60, pop_size=60, verbose=True)
    rules = non_dominated_solutions.get("X").flatten()

    if len(non_dominated_solutions) == 0:
        print("No solutions found")
        return

    # plot the results
    plot = Scatter(title="NSGA-II")
    plot.add(non_dominated_solutions.get("F"))
    plot.axis_labels = explainer._problem.get_objective_names()
    _ = plot.show()

    # self._save_generations_video_pymoo(res.history, ".", "counterfactual_evolution")

    # get rules associated to non_dominated solutions
    print("Factual rule:")
    factual_rule = classifier.get_var(changed_rule_index)
    print(factual_rule)
    factual_rule.get_rule().plot_antecedent()

    print("Rules of non-dominated solutions:")
    for i in range(len(rules)):
        rule = rules[i]
        print(rule)
        if i<10:
            rule.get_rule().plot_antecedent()

    if len(rules) > 10:
        print("Some rule plots were not displayed because there are too many rules")


def param_search(num_classes, classifiers, out_path, param_name, num_experiments=11, min_val=0.0, max_val=1.0, test_dataset=None):
    param_vals = np.linspace(min_val, max_val, num_experiments)

    for param_val in param_vals:
        current_path = os.path.join(out_path, f"{param_name}_{param_val:.2f}")
        kwargs = {param_name: param_val}
        try:
            main_benchmark(num_classes, classifiers, out_path=current_path, test_dataset=test_dataset, **kwargs)
        except Exception as e:
            # raise e
            print(f"Error processing {param_name}={param_val:.2f}: {e}")

