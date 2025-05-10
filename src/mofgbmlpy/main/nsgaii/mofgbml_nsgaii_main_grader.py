import re
from pathlib import Path

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
                          #世代数を表示する
                          verbose=True,
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

    args = [
        "--algorithm-id", "3",
        "--experiment-id", "1",
        "--train-file", "None",
        "--test-file", "None",
        "--data-name", "vehicle_grader",
        "--terminate-evaluation", "180000",
        "--objectives", "num-rules", "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        "--crossover-type", "hybrid-gbml-crossover",
        "--verbose",
    ]

    data_name = "vehicle"
    test_dir = f"C:/Users/Ayato Tomofuji/Documents/Mof/result_grader/{data_name}"
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
    # 1. tstファイルを探索
    test_dir = Path(test_dir)
    experiment_id_index = args.index("--experiment-id") + 1  # "--experiment-id" の次の要素が ID の値
    a_dirs = [d for d in test_dir.iterdir() if d.is_dir() and re.match(r"a\d+_\d+", d.name)]
    print(a_dirs)
    for a_dir in a_dirs:
        train_files = sorted(a_dir.glob("X_masked*.csv"))

        for train_file in train_files:
            # `num_rule` を抽出（例: "X_masked4.csv" → "4"）
            match = re.search(r"X_masked(\d+)\.csv", train_file.name)
            if not match:
                continue  # 形式が合わないファイルは無視

            num_rule = match.group(1)
            test_file = a_dir / f"X_masked{num_rule}_test.csv"

            if not test_file.exists():
                print(f"Warning: Test file {test_file} not found, skipping...")
                continue

            # experiment_id を `a{n}_{m}_{num_rule}` に設定
            experiment_id = f"{a_dir.name}_{num_rule}"
            args[experiment_id_index] = experiment_id

            print(f"Processing Train: {train_file} | Test: {test_file} | Experiment ID: {experiment_id}")

            # `train_file` と `test_file` を処理
            train_set = Input().input_data_set(str(train_file), False)
            test_set = Input().input_data_set(str(test_file), False)
            runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
            results = runner.main(args, train=train_set, test=test_set)

            # 結果の処理
            Xs = results.opt.get("X")[:, 0]
            num_rules = [len(sol.get_vars()) for sol in Xs]

            # 解析結果の書き込み（追記）
            #with open("result_vowel.txt", "a") as f:
            #    f.write(f"{experiment_id}, {train_file}, {results.exec_time}, {num_rules}\n")

