import re
from pathlib import Path

import numpy as np
from pymoo.termination import get_termination
import argparse
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

from mofgbmlpy.data.input import Input
from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_args import MoFGBMLNSGAIIArgs
import sys

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling


class MoFGBMLNSGAIIMain(AbstractMoFGBMLMain):
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
    import os

    os.chdir("C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/")


    data_name = "vowel"


    parser = argparse.ArgumentParser(description="Run MoFGBML experiment with a specific identifier.")
    parser.add_argument("--identifier", type=str, default=None, help="Specify the identifier (e.g., 'a0_0'). If not set, process all files.")
    argv = parser.parse_args()

    args = [
        "--algorithm-id", "1",
        "--experiment-id", f"{argv.identifier}",
        "--train-file", f"dataset/{data_name}/a0_6_{data_name}-10tra.dat",
        "--test-file", f"dataset/{data_name}/a0_6_{data_name}-10tst.dat",
        "--data-name", f"{data_name}",
        "--terminate-evaluation", "180000",
        "--objectives", "num-rules", "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        "--crossover-type", "hybrid-gbml-crossover",
        "--verbose",

    ]



    #for文で，trainとtestのデータをtっ婚で，10-fold CVを複数回行える
    #ここで，dataset_nodes/data_name/の中にある全csvファイルについて再帰的に
    #探索し，それぞれでrunner.mainを実行する．で，テストもまたそれぞれ

    """
    tstファイルを探索し、それに基づいて対応するtraファイルを別ディレクトリから取得して処理する。

    Args:
        test_dir (str): tstファイルが保存されているディレクトリ
        train_base_dir (str): traファイルが保存されているディレクトリのベースパス
        data_name (str): 対象データセット名 (例: "bupa")
    """
   
    print(args[5])
    print(data_name)
    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    results = runner.main(args)
    Xs = results.opt.get("X")[:, 0]
    num_rules = [len(sol.get_vars()) for sol in Xs]
    #plot = runner.get_pareto_front_plot(results.opt)
    #plot.show()
    ## plot.ax.set_ylim([0,1])
    #plot.ax.grid(visible=True)
    results.opt.get('X')[1, 0]
    #各plotのタイトルは，各traファイルの名前に対応するようにする
    #runner.plot_line_interpretability_error_rate_tradeoff(Xs,
    #                                                  file_path=num_rules_path, xlim=[0, 30], x_key='num_rules')
    #各識別器の識別精度を取得
    ##  最適解の中の全ての識別器についてループ
    #for idx, sol in enumerate(results.opt.get("X")[:, 0]):
    #    print(f"\n識別器 {idx + 1} のルール:")
    #    # 各識別器のルールを取得し表示
    #    for rule_idx, var in enumerate(sol.get_vars(), start=1):
    #        print(f"  ルール {rule_idx}: {var.get_rule().get_linguistic_representation()}")
    # 各セットにおいて，s0_0などのセット番号と，そのセットにおけるexec_time(訓練)，そのセットにおける識別器の数，そして書く識別器のルール長を取得し，
    # それをファイルに書き込む，ファイルは1つのファイルで，どんどん追記していく
  
