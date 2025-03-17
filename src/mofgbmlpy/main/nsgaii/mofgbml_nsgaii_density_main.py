import datetime
from pathlib import Path
import re

import numpy as np
from matplotlib import pyplot as plt

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation

from mofgbmlpy.gbml.operator.repair.pittsburgh_repair import PittsburghRepair

from mofgbmlpy.data.input import Input
from mofgbmlpy.data.input_density import Input_density
from mofgbmlpy.main.abstract_mofgbml_density_main import AbstractMoFGBMLDensityMain
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_args import MoFGBMLNSGAIIArgs

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling
import gc

class MoFGBMLNSGAIIDensityMain(AbstractMoFGBMLDensityMain):
    """MoFBML runner for NSGA-II"""
    def __init__(self, knowledge_factory_class):
        """Constructor
        Args:
            knowledge_factory_class (AbstractKnowledgeFactory): Knowledge factory class
        """
        super().__init__(MoFGBMLNSGAIIArgs(), knowledge_factory_class)

    def run(self, dataset_manager):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """
        self.algorithm = NSGA2(pop_size=self._mofgbml_args.get("POPULATION_SIZE"),
                          sampling=HybridGBMLSampling(self._learner),
                          crossover=self._crossover,
                          repair=PittsburghRepair(),
                          mutation=PittsburghMutation(self._knowledge, self._random_gen),
                          eliminate_duplicates=False,
                          save_history=True,
                          n_offsprings=self._mofgbml_args.get("OFFSPRING_POPULATION_SIZE"))
        dataset_manager.set_algorithm(self.algorithm)
        self.res = minimize(self._problem,
                       self.algorithm,
                       copy_algorithm=False,
                       termination=self._termination,
                       seed=self._mofgbml_args.get("RAND_SEED"),
                       verbose=self._verbose)

        return self.res

    def set_train(self, train):
        self.train = train



if __name__ == '__main__':
    #runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    #runner.main(sys.argv[1:])
    import os
    os.chdir("C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/")

    args = [
        "--algorithm-id", "2",
        "--experiment-id", "2",
        "--train-file", "None",
        "--test-file", "None",
        "--data-name", "vehicle_density_adapt",
        "--terminate-evaluation", "180000",
        "--objectives", "num-rules", "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        "--crossover-type", "hybrid-gbml-crossover",
    ]

    data_name = "vehicle"



    minCIM = 0.5
    train_dir = f"art_without_edge/dataset_nodes/{data_name}/"
    test_dir = f"dataset/{data_name}/"
    #for文で，trainとtestのデータをtっ婚で，10-fold CVを複数回行える
    #ここで，dataset_nodes/data_name/の中にある全csvファイルについて再帰的に
    #探索し，それぞれでrunner.mainを実行する．で，テストもまたそれぞれ

    """
    tstファイルを探索し、それに基づいて対応するtraファイルを別ディレクトリから取得して処理する。

    Args:
        test_dir (str): tstファイルが保存されているディレクトリ
        train_base_dir (str): traファイルが保存されているディレクトリのベースパス
        data_name (str): 対象データセット名 (例: "vehicle")
    """
    # 1. tstファイルを探索
    test_dir = Path(test_dir)
    train_base_dir = Path(train_dir)
    experiment_id = 1
    for test_file in test_dir.glob(f"*{data_name}-10tst.dat"):
        args[3] = str(experiment_id)
        # tstファイル名から識別子を抽出 (例: "a0_0_vehicle")
        identifier = test_file.stem.split(f"-10tst")[0]
        print(f"Processing test file: {test_file}")

        # 対応する tra ファイルを含むディレクトリを決定
        train_dir = train_base_dir / f"{identifier}_tra"
        if not train_dir.exists():
            print(f"Train directory not found: {train_dir}")
            continue
        train_files = sorted(train_dir.glob(f"{identifier}_node*.csv"), reverse=True)
        print(train_files)

        train_datasets = [Input_density().input_data_set(train_file, False) for train_file in train_files]
        # 2. traファイルを探索 (例: "a0_0_vehicle_node*.csv")



        test_set = Input().input_data_set(test_file, False)
        runner = MoFGBMLNSGAIIDensityMain(HomoTriangleKnowledgeFactory_2_3_4_5)
        results = runner.main(args, trains=train_datasets, test=test_set)
        Xs = results.opt.get("X")[:, 0]
        num_rules = [len(sol.get_vars()) for sol in Xs]

        #plot = runner.get_pareto_front_plot(results.opt)
        #plot.show()
        ## plot.ax.set_ylim([0,1])
        #plot.ax.grid(visible=True)

        #各plotのタイトルは，各traファイルの名前に対応するようにする
        num_rules_path = f"art_without_edge/result_nodes/adapt/{data_name}/{identifier}_adapt_recip.png"
        title = f"MoFGBMLPy with Density adaptive {identifier} with NSGA-II"
        runner.plot_line_interpretability_error_rate_tradeoff(Xs,
                                                          file_path=num_rules_path, xlim=[0, 20], x_key='num_rules')
        objectives = list(np.unique(results.opt.get("F")))

        ##  最適解の中の全ての識別器についてループ
        #for idx, sol in enumerate(results.opt.get("X")[:, 0]):
        #    print(f"\n識別器 {idx + 1} のルール:")
        #    # 各識別器のルールを取得し表示
        #    for rule_idx, var in enumerate(sol.get_vars(), start=1):
        #        print(f"  ルール {rule_idx}: {var.get_rule().get_linguistic_representation()}")
        # 各セットにおいて，s0_0などのセット番号と，そのセットにおけるexec_time(訓練)，そのセットにおける識別器の数，そして書く識別器のルール長を取得し，
        # それをファイルに書き込む，ファイルは1つのファイルで，どんどん追記していく
        #with open("result_vehicle_density_adapt.txt", "a") as f:
        #    f.write(f"{identifier}, {results.exec_time}, {num_rules}, {objectives} \n")
        #現在の時刻を取得
        now = datetime.datetime.now()
        print(now)
        experiment_id += 1
        del runner, results, train_datasets
        gc.collect()

