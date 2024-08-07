import cProfile
import os
import sys
from datetime import datetime

new_path = f"{os.getcwd()}{os.sep}src"
if new_path not in sys.path:
    sys.path.append(new_path)

from src.mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain
from src.mofgbmlpy.main.moead.mofgbml_moead_main import MoFGBMLMOEADMain
from mofgbmlpy.data.output import Output
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5


def run_profiler(mofgbml_class_name):
    args = [
        "--data-name", "iris",
        "--algorithm-id", "1",
        "--experiment-id", "2",
        # "--num-parallel-cores", "1",
        "--train-file", "dataset/iris/a0_0_iris-10tra.dat",
        "--test-file", "dataset/iris/a0_0_iris-10tst.dat",
        "--terminate-evaluation", "1000",
        "--objectives", "total-rule-length", "error-rate"
    ]

    # args = [
    #     # "--data-name", "flags",
    #     # "--data-name", "richromatic",
    #     "--data-name", "german",
    #     "--algorithm-id", "1",
    #     "--experiment-id", "2",
    #     "--rand-seed", "2020",
    #     # "--train-file", "../dataset/flags/a0_0_flags-10tra.dat",
    #     # "--test-file", "../dataset/flags/a0_0_flags-10tst.dat",
    #     # "--train-file", "../dataset/richromatic/a0_0_richromatic-10tra.dat",
    #     # "--test-file", "../dataset/richromatic/a0_0_richromatic-10tst.dat",
    #     "--train-file", "dataset/german/a0_0_german-10tra.dat",
    #     "--test-file", "dataset/german/a0_0_german-10tst.dat",
    #     "--terminate-evaluation", "200",
    #     "--objectives", "num-rules", "error-rate",
    #     "--is-multi-label",
    #     "--gen-plot",
    #     "--verbose",
    # ]

    cProfile.runctx(f"{mofgbml_class_name}(HomoTriangleKnowledgeFactory_2_3_4_5).main(args)", globals(), locals(), "Profile.pstats")
    os.system("gprof2dot -f pstats Profile.pstats -o Profile.dot -n 0.3 --color-nodes-by-selftime --node-label=self-time-percentage --node-label=total-time --node-label=total-time-percentage")

    profiler_results_folder = "profiler_results"
    Output.mkdirs(profiler_results_folder)
    profile_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

    os.system(f"dot Profile.dot -Tpng -o {profiler_results_folder}/{profile_file_name}")
    os.remove("Profile.pstats")
    os.remove("Profile.dot")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("1 argument expected (MoFGBML class name), e.g. MoFGBMLNSGAIIMain")
    run_profiler(str(sys.argv[1]))

