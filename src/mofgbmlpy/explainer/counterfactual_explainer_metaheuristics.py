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
    def __init__(self, classifier, changed_rule_index, target_class, test_set, mutation_fs_type_prob=0.5, sampling_noise_str=0.1, mutation_prob=0.7, mutated_param_prob=0.6, mutation_revert_to_initial_prob=0.0, crossover_prob=0.7, crossover_p1_selected_prob=0.5, sampling_fs_type_prob=0.0, sampling_change_fs_params_prob=1.0, use_search_space_crowding=False, objectives=["confidence_loss", "change_loss"], n_gen=60, pop_size=60):
        self._problem = CounterfactualProblem(classifier, changed_rule_index, target_class, test_set=test_set, objectives=objectives)

        self._sampling = FuzzySetsSampling(sampling_noise_str, sampling_fs_type_prob, sampling_change_fs_params_prob)
        self._mutation = FuzzySetsMutation(mutation_prob, mutated_param_prob, mutation_fs_type_prob, mutation_revert_to_initial_prob)
        self._crossover = FuzzySetsCrossover(crossover_prob, crossover_p1_selected_prob)
        self._eliminate_duplicates = FuzzySetsEliminateDuplicates(self._problem)
        self._survival = FuzzySetsSurvival(use_search_space_crowding=use_search_space_crowding)
        self._n_gen = n_gen
        self._pop_size = pop_size

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

    def train(self, verbose=True):
        # self._problem.get_fuzzy_rule().plot_antecedent()

        termination = get_termination("n_gen", self._n_gen)

        algorithm = NSGA2(
            pop_size=self._pop_size,
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

    def get_problem(self):
        return self._problem
