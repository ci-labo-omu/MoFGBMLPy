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
from mofgbmlpy.explainer.util import get_config
#TODO: change individual type to Rule instead of FuzzySet

class CounterFactualExplainerMetaheuristics:
    def __init__(self, classifier, changed_rule_index, target_class, mutation_fs_type_prob):
        self._problem = CounterfactualProblem(classifier, changed_rule_index, target_class)

        self._sampling = FuzzySetsSampling()
        self._mutation = FuzzySetsMutation(0.7, 0.6, prob_change_type=mutation_fs_type_prob)
        self._crossover = FuzzySetsCrossover(0.7, 0.5)
        self._eliminate_duplicates = FuzzySetsEliminateDuplicates(self._problem)
        self._survival = FuzzySetsSurvival(self._eliminate_duplicates)

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
            eliminate_duplicates=self._eliminate_duplicates,
            save_history=False,  # True,
            survival=self._survival,
        )

        res = minimize(self._problem, algorithm, seed=41, verbose=verbose, termination=termination)

        non_dominated_solutions = res.opt

        non_dominated_solutions = self.remove_non_target_class_solutions(non_dominated_solutions, self.get_target_class())
        rules = non_dominated_solutions.get("X").flatten()

        return non_dominated_solutions, rules


def main_benchmark(dataset, non_dominated_solutions, out_path, mutation_fs_type_prob=0.5):
    times = []
    num_sols = []
    ious = []
    conf = []
    num_failures = 0
    num_runs = 0

    class_labels = [ClassLabelBasic(c) for c in range(dataset.get_num_classes())]

    for p_sol in tqdm(non_dominated_solutions):
        num_rules = p_sol[0].get_num_vars()
        for i_var in range(num_rules):
            class_label = p_sol[0].get_var(i_var).get_class_label()
            for target_class in class_labels:
                if target_class == class_label:
                    continue
                start = time.time()

                try:
                    explainer = CounterFactualExplainerMetaheuristics(p_sol[0], i_var, target_class, mutation_fs_type_prob=mutation_fs_type_prob)
                    non_dominated_solutions, rules = explainer.train(n_gen=60, pop_size=60, verbose=False)
                    # if len(non_dominated_solutions) == 0:
                    #     raise ValueError("No solutions found")
                except Exception as e:
                    raise e
                    non_dominated_solutions = np.array([], dtype=object)
                    rules = np.array([], dtype=object)
                    start = None

                end = time.time()

                if start is not None and len(non_dominated_solutions) > 0:
                    times.append(end - start)
                    num_sols.append(len(non_dominated_solutions))
                    ious.append(
                        (np.min(1-non_dominated_solutions.get("F")[:,1]), np.max(1-non_dominated_solutions.get("F")[:,1]))
                    )
                    conf.append(
                        (np.min(1-non_dominated_solutions.get("F")[:,0]), np.max(1-non_dominated_solutions.get("F")[:,0]))
                    )


                    # # plot the results
                    # plot = Scatter(title="NSGA-II")
                    # plot.add(non_dominated_solutions.get("F"))
                    # plot.axis_labels = explainer._problem.get_objective_names()
                    # _ = plot.show()

                else:
                    num_failures += 1
                num_runs += 1

    df = pd.DataFrame({
        "time": times,
        "num_sols": num_sols,
        "iou_min": [x[0] for x in ious],
        "iou_max": [x[1] for x in ious],
        "conf_min": [x[0] for x in conf],
        "conf_max": [x[1] for x in conf],
    })

    os.makedirs(out_path, exist_ok=False)
    df.to_csv(f"{out_path}\\results.csv", index=False)

    with open(f"{out_path}\\results_summary.txt", "w") as f:
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Number of failures: {num_failures}\n")

        if len(times) != 0:
            f.write(f"Median time: {np.median(times):.2f} seconds\n")
            f.write(f"Median number of solutions: {np.median(num_sols):.2f}\n")
            f.write(f"Min IOU: {np.min([x[0] for x in ious]):.2f}\n")
            f.write(f"Max IOU: {np.max([x[1] for x in ious]):.2f}\n")
            f.write(f"Min confidence: {np.min([x[0] for x in conf]):.2f}\n")
            f.write(f"Max confidence: {np.max([x[1] for x in conf]):.2f}\n")

def main_plot_single(dataset, non_dominated_solutions, mutation_fs_type_prob=0.5):
    classifier = non_dominated_solutions[0][0]
    changed_rule_index = 0

    target_class = ClassLabelBasic(0)
    explainer = CounterFactualExplainerMetaheuristics(classifier, changed_rule_index, target_class, mutation_fs_type_prob)
    non_dominated_solutions, rules = explainer.train(n_gen=60, pop_size=60, verbose=True)

    # plot the results
    plot = Scatter(title="NSGA-II")
    plot.add(non_dominated_solutions.get("F"))
    plot.axis_labels = explainer._problem.get_objective_names()
    _ = plot.show()

    # self._save_generations_video_pymoo(res.history, ".", "counterfactual_evolution")

    # get rules associated to non_dominated solutions
    print("Rules of non-dominated solutions:")
    for rule in rules:
        print(rule)

    explainer._problem.get_fuzzy_rule().plot_antecedent()
    if len(rules) != 0:
        for i in range(len(rules)):
            rules[i].plot_antecedent()
            #         print(rules[i].get_knowledge())
            #         print(rules[i].get_knowledge().get_fuzzy_set(6, 1).get_function().get_params())

            antecedent_indices = rules[i].get_antecedent().get_antecedent_indices()
            fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
            for j, idx in enumerate(antecedent_indices):
                fuzzy_sets[j] = rules[i].get_knowledge().get_fuzzy_set(j, idx)
            current_mf_values = explainer._problem.compute_membership_values(fuzzy_sets, 0, 1)

            iou = explainer._problem.compute_iou(
                explainer._problem.get_initial_mfs_y(), current_mf_values, step=1 / current_mf_values.shape[1]
            )

            confidences = explainer._problem._learner.calc_confidence_py(rules[i].get_antecedent(), dataset)

            print(f"Rule {i}: {np.mean(iou):.3f} and Confidence: {confidences[target_class.get_class_label_value()]:.3f}")


def mutation_param_search(data_name, out_path, num_experiments=11):
    param_vals = np.linspace(0, 1, num_experiments)
    dataset, non_dominated_solutions = get_config(data_name)

    for param_val in param_vals:
        current_path = os.path.join(out_path, f"mut_{param_val:.2f}")
        main_benchmark(dataset, non_dominated_solutions, out_path=current_path, mutation_fs_type_prob=param_val)


if __name__ == "__main__":
    # dataset, non_dominated_solutions = get_config("pima")
    # main_plot_single(dataset, non_dominated_solutions)

    for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima","sonar", "spectfheart", "tae", "wisconsin"]:
        result_path = f"..\\..\\..\\cf_results_v2\\cf_metaheuristics\\{data_name}"
        dataset, non_dominated_solutions = get_config(data_name)
        main_benchmark(dataset, non_dominated_solutions, out_path=result_path)

    # mutation_param_search("iris", out_path="..\\..\\..\\cf_results\\cf_metaheuristics_mutation_param_search\\iris", num_experiments=11)
