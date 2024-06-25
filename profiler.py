import cProfile
import os
import sys
from datetime import datetime

new_path = f"{os.getcwd()}{os.sep}src"
if new_path not in sys.path:
    sys.path.append(new_path)

from src.mofgbmlpy.main.basic.mofgbml_basic_main import MoFGBMLBasicMain
from src.mofgbmlpy.data.output import Output

def main():
    args = [
        "iris",
        1,
        2,
        1,
        "dataset/iris/a0_0_iris-10tra.dat",
        "dataset/iris/a0_0_iris-10tst.dat",
    ]

    cProfile.runctx("MoFGBMLBasicMain.main(args)", globals(), locals(), "Profile.pstats")
    os.system("gprof2dot -f pstats Profile.pstats -o Profile.dot -n 0.3 --color-nodes-by-selftime --node-label=self-time-percentage --node-label=total-time --node-label=total-time-percentage")

    profiler_results_folder = "profiler_results"
    Output.mkdirs(profiler_results_folder)
    profile_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

    os.system(f"dot Profile.dot -Tpng -o {profiler_results_folder}/{profile_file_name}")
    os.remove("Profile.pstats")
    os.remove("Profile.dot")


if __name__ == "__main__":
    main()
