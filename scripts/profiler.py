import cProfile
import os
import sys
from datetime import datetime

from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics import main_plot_single
from scripts.cf_explanation_tests import get_config

new_path = f"{os.getcwd()}{os.sep}src"
if new_path not in sys.path:
    sys.path.append(new_path)

from mofgbmlpy.data.output import Output
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5


def generate_plot(profiler_results_folder):
    os.system(
        "gprof2dot -f pstats Profile.pstats -o Profile.dot -n 0.3 --color-nodes-by-selftime --node-label=self-time-percentage --node-label=total-time --node-label=total-time-percentage")

    Output.mkdirs(profiler_results_folder)
    profile_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

    os.system(f"dot Profile.dot -Tpng -o {profiler_results_folder}/{profile_file_name}")
    os.remove("Profile.pstats")
    os.remove("Profile.dot")


def run_profiler(mofgbml_class_name):
    args = [
        "--data-name", "iris",
        "--algorithm-id", "1",
        "--experiment-id", "2",
        # "--num-parallel-cores", "1",
        "--train-file", "../dataset/iris/a0_0_iris-10tra.dat",
        "--test-file", "../dataset/iris/a0_0_iris-10tst.dat",
        "--terminate-evaluation", "1000",
        "--objectives", "total-rule-length", "error-rate"
    ]

    cProfile.runctx(f"{mofgbml_class_name}(HomoTriangleKnowledgeFactory_2_3_4_5).main(args)", globals(), locals(), "Profile.pstats")

    profiler_results_folder = "../profiler_results"
    generate_plot(profiler_results_folder)


def run_profiler_cf_explainer():
    _, non_dominated_solutions = get_config("pima")

    cProfile.runctx(f"main_plot_single(non_dominated_solutions)", globals(), locals(),
                    "Profile.pstats")

    profiler_results_folder = "../profiler_results/cf_profiler_results"
    generate_plot(profiler_results_folder)


if __name__ == "__main__":
    run_profiler_cf_explainer()
    # if len(sys.argv) != 2:
    #     raise Exception("1 argument expected (MoFGBML class name), e.g. MoFGBMLNSGAIIMain")
    # run_profiler(str(sys.argv[1]))

