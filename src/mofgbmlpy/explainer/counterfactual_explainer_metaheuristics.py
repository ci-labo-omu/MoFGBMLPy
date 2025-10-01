import random
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.optimize import minimize
from tqdm import tqdm

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


class CounterFactualExplainerMetaheuristics:
    def __init__(self, classifier, changed_rule_index, target_class, test_set, mutation_fs_type_prob=0.5, sampling_noise_str=0.1, mutation_prob=0.7, mutated_param_prob=0.6, mutation_revert_to_initial_prob=0.0, crossover_prob=0.7, crossover_p1_selected_prob=0.5, sampling_fs_type_prob=0.0, sampling_change_fs_params_prob=1.0, use_search_space_crowding=False, objectives=["confidence_loss", "change_loss"]):
        self._problem = CounterfactualProblem(classifier, changed_rule_index, target_class, test_set=test_set, objectives=objectives)

        self._sampling = FuzzySetsSampling(sampling_noise_str, sampling_fs_type_prob, sampling_change_fs_params_prob)
        self._mutation = FuzzySetsMutation(mutation_prob, mutated_param_prob, mutation_fs_type_prob, mutation_revert_to_initial_prob)
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
            return Population.new(X=np.array([], dtype=object), F=np.array([], dtype=float))

        non_dominated_solutions = self._eliminate_duplicates.do(non_dominated_solutions)

        if len(non_dominated_solutions) == 0:
            return non_dominated_solutions

        non_dominated_solutions = self.remove_non_target_class_solutions(non_dominated_solutions, self.get_target_class())

        return non_dominated_solutions

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

    def all_metrics_eval(self, solutions):
        results = {}
        initial_train_error_rate = self._problem.error_rate(use_test_set=False, initial_classifier=True)
        initial_test_error_rate = self._problem.error_rate(use_test_set=True, initial_classifier=True)
        initial_num_wins_list, initial_num_successes_list = self._problem.num_wins_and_successes(initial_classifier=True)
        rule_index = self._problem.get_changed_rule_index()
        initial_num_wins = initial_num_wins_list[rule_index]
        initial_num_successes = initial_num_successes_list[rule_index]
        initial_num_wins_freq = initial_num_wins / np.sum(initial_num_wins_list)
        initial_num_successes_freq = initial_num_successes / np.sum(initial_num_successes_list)


        for sol in solutions:
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "confidence_loss", self._problem.conf_loss(sol))
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "change_loss", self._problem.change_loss(sol))
            results = CounterFactualExplainerMetaheuristics.__append_metric(results, "num_changed_features", self._problem.num_changed_features_loss(sol))

            for replace in [True, False]:
                for use_test in [True, False]:
                    metric_name = f"{'test' if use_test else 'train'}_error_rate_{'replace' if replace else 'append'}"
                    error_rate_val = self._problem.error_rate(sol, replace=replace, use_test_set=use_test)
                    results = CounterFactualExplainerMetaheuristics.__append_metric(results, metric_name, error_rate_val)
                    init_val = initial_test_error_rate if use_test else initial_train_error_rate
                    var_value = error_rate_val - init_val
                    results = CounterFactualExplainerMetaheuristics.__append_metric(results, f"{metric_name}_initial_variation", var_value)

            for (repl, txt) in [(False, "append"), (True, "replace")]:
                num_wins_list, num_successes_list = self._problem.num_wins_and_successes(sol, replace=replace)
                idx = rule_index if repl else len(num_wins_list) - 1
                for freq in [False, True]:
                    val_num_wins, val_num_successes = num_wins_list[idx], num_successes_list[idx]

                    if freq:
                        val_num_wins /= np.sum(num_wins_list) if np.sum(num_wins_list) > 0 else 0
                        val_num_successes /= np.sum(num_successes_list) if np.sum(num_successes_list) > 0 else 0
                        val_initial_num_wins, val_initial_num_successes = initial_num_wins_freq, initial_num_successes_freq
                        txt += "_freq"
                    else:
                        val_initial_num_wins, val_initial_num_successes = initial_num_wins, initial_num_successes

                    results = CounterFactualExplainerMetaheuristics.__append_metric(results, f"num_wins_{txt}", val_num_wins)
                    results = CounterFactualExplainerMetaheuristics.__append_metric(results, f"num_successes_{txt}", val_num_successes)
                    results = CounterFactualExplainerMetaheuristics.__append_metric(results, f"num_wins_initial_variation_{txt}", val_num_wins - val_initial_num_wins)
                    results = CounterFactualExplainerMetaheuristics.__append_metric(results, f"num_successes_initial_variation_{txt}", val_num_successes - val_initial_num_successes)

        return results


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


def main_benchmark(num_classes, classifiers, out_path, seed=2017, test_dataset=None, data_name="", test_name="", n_gen=60, pop_size=60, **kwargs):
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

    desc = f"Running test {test_name}"
    if data_name != "":
        desc += f" on {data_name}"

    for p_sol in tqdm(classifiers, desc=desc):
        num_rules = p_sol[0].get_num_vars()
        for i_var in range(num_rules):
            class_label = p_sol[0].get_var(i_var).get_class_label()
            for target_class in class_labels:
                if target_class == class_label:
                    continue
                start = time.time()

                explainer = CounterFactualExplainerMetaheuristics(p_sol[0], i_var, target_class, test_dataset, **kwargs)
                solutions = explainer.train(n_gen=n_gen, pop_size=pop_size, verbose=False)

                if solutions is None or len(solutions) == 0:
                    num_failures += 1
                    num_runs += 1
                    continue

                end = time.time()

                current_metrics_vals = explainer.all_metrics_eval(solutions.get("X").flatten())

                metrics_values["time_in_seconds"].append(end - start)
                metrics_values["num_sols"].append(len(solutions))
                metrics_values["diversity"].append(CounterFactualExplainerMetaheuristics.compute_diversity(solutions))

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


def param_search(num_classes, classifiers, out_path, param_name, num_experiments=11, min_val=0.0, max_val=1.0, test_dataset=None, data_name="", is_int=False):
    if is_int:
        param_vals = np.linspace(min_val, max_val, num_experiments, dtype=int)
    else:
        param_vals = np.linspace(min_val, max_val, num_experiments)

    for param_val in param_vals:
        current_path = os.path.join(out_path, f"{param_name}_{param_val:.2f}")
        kwargs = {param_name: param_val}
        # try:
        if os.path.exists(current_path):
            print(f"Path {current_path} already exists (skipped).")
            continue
        test_name = f"Param search {param_name}={param_val:.2f}"
        main_benchmark(num_classes, classifiers, out_path=current_path, test_dataset=test_dataset, data_name=data_name, test_name=test_name, **kwargs)
        # except Exception as e:
        #     print(f"Error processing {param_name}={param_val:.2f}: {e}")

