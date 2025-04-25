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
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
import os
import numpy as np


class CounterFactualExplainerMetaheuristics:
    def __init__(self, fuzzy_rule, target_class, learner):
        initial_knowledge = fuzzy_rule.get_antecedent().get_knowledge()
        initial_class = fuzzy_rule.get_class_label()

        self._problem = CounterfactualProblem(initial_knowledge, fuzzy_rule, initial_class, target_class, learner)
        self._sampling = FuzzySetsSampling()
        self._mutation = FuzzySetsMutation(0.5)
        self._crossover = FuzzySetsCrossover(0.5)

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

    def train(self):
        pop_size = 100
        termination = get_termination("n_gen", 50)

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=self._sampling,
            crossover=self._crossover,
            mutation=self._mutation,  # should consider bounds and conditions of membership functions params
            eliminate_duplicates=False,
            save_history=True,
        )

        res = minimize(self._problem, algorithm, seed=41, verbose=True, termination=termination)

        # plot the results
        plot = Scatter(title="NSGA-II")
        plot.add(res.F)
        plot.axis_labels = self._problem.get_objective_names()
        _ = plot.show()

        # self._save_generations_video_pymoo(res.history, ".", "counterfactual_evolution")

        non_dominated_solutions = res.opt.get("X")

        print("Non-dominated solutions:", res.opt.get("F"))

        # get rules associated to non_dominated solutions
        rules = [self._problem.build_rule(solution) for solution in non_dominated_solutions]
        print("Rules of non-dominated solutions:")
        for rule in rules:
            print(rule)

        # Only keep rules with the target class
        target_rules = [rule for rule in rules if rule.get_class_label() == self._problem.get_target_class()]
        # same with F

        print()
        print("Target rules:")
        for rule in target_rules:
            print(rule)

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

    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    res = runner.main(args)

    non_dominated_solutions = res.X
    objectives_non_dominated_solutions = res.F

    sol1 = non_dominated_solutions[0]
    rule = sol1[0].get_var(0).get_rule()

    learner = LearningBasic(runner.get_train_set())
    target_class = ClassLabelBasic(1)

    explainer = CounterFactualExplainerMetaheuristics(rule, target_class, learner)
    explainer.train()
