import time
from pathlib import Path
import re

import numpy as np
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

from mofgbmlpy.data.input import Input
from mofgbmlpy.data.input_density import Input_density
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
    #runner.main(sys.argv
    # [1:])
    import os
    os.chdir("C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/")

    args = [
        "--algorithm-id", "1",
        "--experiment-id", "2",
        "--train-file", "None",
        "--test-file", "None",
        "--data-name", "iris",
        "--terminate-evaluation", "120000",
        "--objectives", "num-rules", "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        "--crossover-type", "hybrid-gbml-crossover",

    ]

    data_name = "iris"
    minCIM = 0.5
    train_dir = f"art_without_edge/dataset_nodes/{data_name}/"
    test_dir = f"dataset/{data_name}/"
    #for文で，trainとtestのデータをつっこんで，10-fold CVを複数回行える
    #ここで，dataset_nodes/data_name/の中にある全csvファイルについて再帰的に
    #探索し，それぞれでrunner.mainを実行する．で，テストもまたそれぞれ

    """
    tstファイルを探索し、それに基づいて対応するtraファイルを別ディレクトリから取得して処理する。

    Args:
        test_dir (str): tstファイルが保存されているディレクトリ
        train_base_dir (str): traファイルが保存されているディレクトリのベースパス
        data_name (str): 対象データセット名 (例: "iris")
    """
    start = time.time()
    print(f"start: {start}")
    # 1. tstファイルを探索
    test_dir = Path(test_dir)
    train_base_dir = Path(train_dir)

    for test_file in test_dir.glob(f"*{data_name}-10tst.dat"):
        # tstファイル名から識別子を抽出 (例: "a0_0_iris")
        identifier = test_file.stem.split(f"-10tst")[0]
        print(f"Processing test file: {test_file} | Identifier: {identifier}")

        # 対応する tra ファイルを含むディレクトリを決定
        train_dir = train_base_dir / f"{identifier}_tra"
        if not train_dir.exists():
            print(f"Train directory not found: {train_dir}")
            continue

        # 2. traファイルを探索 (例: "a0_0_iris_node*.csv")
        train_files = sorted(train_dir.glob(f"{identifier}_node*.csv"))
        if not train_files:
            print(f"No training files found in: {train_dir}")
            continue

        # 3. traファイルとtstファイルを対応させて処理
        for train_file in train_files:
            print(f"Processing Train: {train_file} | Test: {test_file}")
            # 実際の処理 (例: runner.main を呼び出す)
            train_file = str(train_file)
            test_file = str(test_file)
            # Extract numeric part (e.g., `30` or `45`) from the file name
            match = re.search(r"node(\d+)", Path(train_file).stem)
            node_number = match.group(1) if match else "unknown"

            train_set = Input_density().input_data_set(train_file, False)
            test_set = Input().input_data_set(test_file, False)
            runner = MoFGBMLNSGAIIDensityMain(HomoTriangleKnowledgeFactory_2_3_4_5)
            results = runner.main(args, train=train_set, test=test_set)
            Xs = results.opt.get("X")[:, 0]
            num_rules = [len(sol.get_vars()) for sol in Xs]






            #各plotのタイトルは，各traファイルの名前に対応するようにする
            num_rules_path = f"art_without_edge/result_nodes/{data_name}/{identifier}_node{node_number}.png"
            title = f"MoFGBMLPy with Density {train_file}{int(minCIM*100)} with NSGA-II"

            runner.plot_line_interpretability_error_rate_tradeoff(Xs,
                                                              file_path=num_rules_path, xlim=[0, 10], x_key='num_rules')

            ##  最適解の中の全ての識別器についてループ
            #for idx, sol in enumerate(results.opt.get("X")[:, 0]):
            #    print(f"\n識別器 {idx + 1} のルール:")

            #    # 各識別器のルールを取得し表示
            #    for rule_idx, var in enumerate(sol.get_vars(), start=1):
            #        print(f"  ルール {rule_idx}: {var.get_rule().get_linguistic_representation()}")

            objectives = [[sol.get("F")[0], sol.get("F")[1]] for sol in Xs]
            # 各セットにおいて，s0_0などのセット番号と，そのセットにおけるexec_time(訓練)，そのセットにおける識別器の数，そして書く識別器のルール長を取得し，
            # それをファイルに書き込む，ファイルは1つのファイルで，どんどん追記していく
            with open("results_density/result_iris_density.txt", "a") as f:
                f.write(f"{train_file},{results.exec_time},{objectives} \n")
    end = time.time()
    print(f"end: {end}")