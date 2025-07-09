import cProfile
import os
import sys
from datetime import datetime

from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain

new_path = f"{os.getcwd()}{os.sep}src"
if new_path not in sys.path:
    sys.path.append(new_path)

from mofgbmlpy.data.output import Output


def run_profiler():
    args = [
        "--data-name", "iris",
        "--algorithm-id", "1",
        "--experiment-id", "2",
        # "--num-parallel-cores", "1",
        "--train-file", "dataset/contraceptive/a0_0_contraceptive-10tra.dat",
        "--test-file", "dataset/contraceptive/a0_0_contraceptive-10tst.dat",
        "--terminate-evaluation", "500",
        "--population-size", "60",
        "--offspring-population-size", "60",
        "--objectives", "num-rules", "error-rate",
        "--algorithm", "nsga2",
        "--verbose"
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    cProfile.runctx(f"PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name).run(args)", globals(), locals(), "Profile.pstats")
    os.system("gprof2dot -f pstats Profile.pstats -o Profile.dot -n 0.3 --color-nodes-by-selftime --node-label=self-time-percentage --node-label=total-time --node-label=total-time-percentage")

    profiler_results_folder = "profiler_results"
    Output.mkdirs(profiler_results_folder)
    profile_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

    os.system(f"dot Profile.dot -Tpng -o {profiler_results_folder}/{profile_file_name}")
    os.remove("Profile.pstats")
    os.remove("Profile.dot")


if __name__ == "__main__":
    run_profiler()

