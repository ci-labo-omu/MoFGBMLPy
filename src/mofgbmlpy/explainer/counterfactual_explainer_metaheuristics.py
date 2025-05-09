from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
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


class CounterFactualExplainerMetaheuristics:
    def __init__(self, fuzzy_rule, target_class, learner):
        initial_knowledge = fuzzy_rule.get_antecedent().get_knowledge()
        initial_class = fuzzy_rule.get_class_label()

        self._problem = CounterfactualProblem(initial_knowledge, fuzzy_rule, initial_class, target_class, learner)
        self._sampling = FuzzySetsSampling()
        self._mutation = FuzzySetsMutation(0.7, 0.6, prob_change_type=0.5)
        self._crossover = FuzzySetsCrossover(0.7, 0.5)
        self._eliminate_duplicates = FuzzySetsEliminateDuplicates(self._problem)
        self._survival = FuzzySetsSurvival(self._eliminate_duplicates)

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
            # for each algorithm object in the history
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

                # finally record the current visualization to the video
                rec.record()

    def train(self, n_gen=100):
        # self._problem.get_fuzzy_rule().plot_antecedent()
        # print(f"END antecedent: {self._problem.get_fuzzy_rule().get_antecedent()}")
        # print(f"END consequent: {self._problem.get_fuzzy_rule().get_consequent()}")

        pop_size = 100
        termination = get_termination("n_gen", n_gen)

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=self._sampling,
            crossover=self._crossover,
            mutation=self._mutation,  # should consider bounds and conditions of membership functions params
            eliminate_duplicates=self._eliminate_duplicates,
            save_history=True,
            survival=self._survival,
        )

        res = minimize(self._problem, algorithm, seed=41, verbose=True, termination=termination)

        # plot the results
        plot = Scatter(title="NSGA-II")
        plot.add(res.F)
        plot.axis_labels = self._problem.get_objective_names()
        _ = plot.show()

        # self._save_generations_video_pymoo(res.history, ".", "counterfactual_evolution")

        non_dominated_solutions = res.opt.get("X")

        # print("Non-dominated solutions:", res.opt.get("F"))

        # get rules associated to non_dominated solutions
        rules = [self._problem.build_rule(solution) for solution in non_dominated_solutions]
        # print("Rules of non-dominated solutions:")
        # for rule in rules:
        #     print(rule)

        # Only keep rules with the target class
        target_rules = [rule for rule in rules if rule.get_class_label() == self._problem.get_target_class()]

        print()
        print("Target rules:")
        for rule in target_rules:
            print(rule)

        self._problem.get_fuzzy_rule().plot_antecedent()
        if len(target_rules) != 0:
            for i in range(len(target_rules)):
                target_rules[i].plot_antecedent()
                #         print(target_rules[i].get_knowledge())
                #         print(target_rules[i].get_knowledge().get_fuzzy_set(6, 1).get_function().get_params())

                antecedent_indices = target_rules[i].get_antecedent().get_antecedent_indices()
                fuzzy_sets = np.empty(len(antecedent_indices), dtype=object)
                for j, idx in enumerate(antecedent_indices):
                    fuzzy_sets[j] = target_rules[i].get_knowledge().get_fuzzy_set(j, idx)
                current_mf_values = self._problem.compute_membership_values(fuzzy_sets, 0, 1)

                iou = self._problem.compute_iou(
                    self._problem.get_initial_mfs_y(), current_mf_values, step=1 / current_mf_values.shape[1]
                )
                print(f"IoU rule {i}: {np.mean(iou):.3f}")

        # # rule with highest confidence
        # rule = max(rules, key=lambda r: self._problem._learner.calc_confidence_py(
        # r.get_antecedent(),
        # self._problem._train_set
        # )[1])
        # print("Rule with highest confidence:", rule)
        #
        # # print confidence of rule with highest confidence
        # confidences = self._problem._learner.calc_confidence_py(rule.get_antecedent(), self._problem._train_set)
        # print([c for c in confidences])
        # # for p in self._problem._learner.get_training_set().get_patterns():
        # p = self._problem._learner.get_training_set().get_patterns()[8]
        # print(p)
        #
        # # print compatibility grade with current pattern
        # fitness_val = rule.get_fitness_value(p.get_attributes_vector())
        # print(f"Fitness value: {fitness_val:.3f}")
        #
        # compatibility_grade = rule.get_antecedent().get_compatible_grade_value_py(p.get_attributes_vector())
        # print(f"Compatibility grade: {compatibility_grade:.3f}")
        #
        # antecedent_indices = rule.get_antecedent().get_antecedent_indices()
        # for i, idx in enumerate(antecedent_indices):
        #     mf_val = rule.get_knowledge().get_membership_value_py(p.get_attributes_vector()[i], i, idx)
        #     print(f"Membership function value: {mf_val:.3f}")
        #
        # print(rule.get_knowledge().get_fuzzy_set(1, antecedent_indices[1]).get_function().get_params())
        #
        # for p in self._problem._train_set.get_patterns():
        #     # if fitness > 0 then print
        #     if rule.get_fitness_value(p.get_attributes_vector()) > 0:
        #         print(p)
        #         print(rule.get_antecedent().get_compatible_grade_value_py(p.get_attributes_vector()))

        return res


if __name__ == "__main__":
    args = [
        "--data-name",
        "appendicitis",
        "--algorithm-id",
        "0",
        "--experiment-id",
        "0",
        "--train-file",
        "..\\..\\..\\dataset\\appendicitis\\a0_0_appendicitis-10tra.dat",
        "--test-file",
        "..\\..\\..\\dataset\\appendicitis\\a0_0_appendicitis-10tra.dat",
        "--terminate-evaluation",
        "1000",
        "--no-output-files",
        "--objectives",
        "error-rate",
        "num-rules",
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
    res = runner.run(args)

    non_dominated_solutions = res.X
    objectives_non_dominated_solutions = res.F

    sol1 = non_dominated_solutions[0]
    rule = sol1[0].get_var(0).get_rule()

    learner = LearningBasic(runner.get_train_set())
    target_class = ClassLabelBasic(1)

    import time

    start = time.time()

    explainer = CounterFactualExplainerMetaheuristics(rule, target_class, learner)
    explainer.train()

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
