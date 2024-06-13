from src.main.arguments import Arguments
from src.data.output import Output
import os


class MoFGBMLBasicArgs(Arguments):

    def load(self, args):
        if len(args) < 6:
            raise Exception("Not enough arguments (6 were expected)")

        self.set("DATA_NAME", args[0])
        self.set("ALGORITHM_ID", args[1])
        self.set("ALGORITHM_ID_DIR", str(os.path.join(self.get("ROOT_FOLDER"), str(self.get("ALGORITHM_ID")))))

        Output.mkdirs(self.get("ALGORITHM_ID_DIR"))

        self.set("EXPERIMENT_ID", args[2])
        self.set("EXPERIMENT_ID_DIR",
                 str(os.path.join(
                     self.get("ALGORITHM_ID_DIR"),
                     self.get("DATA_NAME"),
                     str(self.get("EXPERIMENT_ID")))))
        Output.mkdirs(self.get("EXPERIMENT_ID_DIR"))

        self.set("NUM_PARALLEL_CORES", int(args[3]))
        self.set("TRAIN_FILE", args[4])
        self.set("TEST_FILE", args[5])
        self.set("IS_MULTI_LABEL", False)
