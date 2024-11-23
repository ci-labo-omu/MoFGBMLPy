from pymoo.termination import get_termination

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory

from mofgbmlpy.gbml.operator.repair.pittsburgh_repair import PittsburghRepair
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_5 import HomoTriangleKnowledgeFactory_5

from mofgbmlpy.main.abstract_mofgbml_density_main import AbstractMoFGBMLDensityMain
from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_args import MoFGBMLNSGAIIArgs
import sys

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling


class MoFGBMLNSGAIIDensityMain(AbstractMoFGBMLDensityMain):
    """MoFBML runner for NSGA-II"""
    def __init__(self, knowledge_factory_class):
        """Constructor
        Args:
            knowledge_factory_class (AbstractKnowledgeFactory): Knowledge factory class
        """
        super().__init__(MoFGBMLNSGAIIArgs(), knowledge_factory_class)

    def run(self):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """
        algorithm = NSGA2(pop_size=self._mofgbml_args.get("POPULATION_SIZE"),
                          sampling=HybridGBMLSampling(self._learner),
                          crossover=self._crossover,
                          repair=PittsburghRepair(),
                          mutation=PittsburghMutation(self._knowledge, self._random_gen),
                          eliminate_duplicates=False,
                          save_history=True,
                          n_offsprings=self._mofgbml_args.get("OFFSPRING_POPULATION_SIZE"))

        res = minimize(self._problem,
                       algorithm,
                       termination=self._termination,
                       seed=self._mofgbml_args.get("RAND_SEED"),
                       verbose=self._verbose)
        return res


if __name__ == '__main__':
    #runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    #runner.main(sys.argv[1:])
    import os
    os.chdir("C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/examples")

    args = [
        "--algorithm-id", "1",
        "--experiment-id", "2",
        "--data-name", "pima",
        "--train-file", f"../art_without_edge/dataset_nodes/satimage/a0_0_satimage_tra/a0_0_satimage_node50.csv",
        "--test-file", "../dataset/satimage/a0_0_satimage-10tst.dat",
        "--terminate-evaluation", "10000",
        "--objectives", "total-rule-length", "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        "--crossover-type", "hybrid-gbml-crossover",
        "--verbose",
    ]
    data_name = "satimage"
    minCIM = 0.5
    runner = MoFGBMLNSGAIIDensityMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    results = runner.main(args)


    min_length = results.opt.get("X")[0, 0].get_var(0).get_rule().get_length()
    max_length = min_length
    for sol in results.opt.get("X")[:, 0]:
        for var in sol.get_vars():
            length = var.get_rule().get_length()
            if length < min_length:
                min_length = length
            elif length > max_length:
                max_length = length

    i = 1
    for var in results.opt.get("X")[0, 0].get_vars():
        print(f"{i}:\t{var.get_rule().get_linguistic_representation()}")
        i += 1

    plot = runner.get_pareto_front_plot(results.opt)
    plot.show()
    # plot.ax.set_ylim([0,1])
    plot.ax.grid(visible=True)
    results.opt.get('X')[1, 0]
    runner.plot_line_interpretability_error_rate_tradeoff(results.opt.get('X')[:, 0],
                                                          title=f"MoFGBMLPy Density3 {str(data_name)}{int(minCIM*100)} with NSGA-II", xlim=[0, 51])
    runner.plot_line_interpretability_error_rate_tradeoff(results.opt.get('X')[:, 0],
                                                          title=f"MoFGBMLPy Density3 {str(data_name)}{int(minCIM*100)} with NSGA-II", xlim=[0, 51], x_key='num_rules')

    print(results.opt.get('F'))
    #  最適解の中の全ての識別器についてループ
    for idx, sol in enumerate(results.opt.get("X")[:, 0]):
        print(f"\n識別器 {idx + 1} のルール:")

        # 各識別器のルールを取得し表示
        for rule_idx, var in enumerate(sol.get_vars(), start=1):
            print(f"  ルール {rule_idx}: {var.get_rule().get_linguistic_representation()}")
