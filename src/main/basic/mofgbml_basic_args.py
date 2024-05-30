from src.main.consts import Consts
from src.utility.Output import Output
import os


class MoFGBMLBasicArgs:
    __data_name = None
    __algorithm_id = None
    __experiment_id = None
    __num_parallel_cores = None
    __train_file = None
    __test_file = None

    @staticmethod
    def load(args):
        if len(args) < 6:
            raise Exception("Not enough arguments (6 were expected)")

        MoFGBMLBasicArgs.__data_name = args[0]

        MoFGBMLBasicArgs.__algorithm_id = args[1]
        Consts.ALGORITHM_ID_DIR = str(os.path.join(Consts.ROOTFOLDER, str(MoFGBMLBasicArgs.__algorithm_id)))
        Output.mkdirs(Consts.ALGORITHM_ID_DIR)

        MoFGBMLBasicArgs.__experiment_id = args[2]
        Consts.EXPERIMENT_ID_DIR = str(os.path.join(Consts.ALGORITHM_ID_DIR, MoFGBMLBasicArgs.__data_name, str(MoFGBMLBasicArgs.__experiment_id)))
        Output.mkdirs(Consts.EXPERIMENT_ID_DIR)

        MoFGBMLBasicArgs.__num_parallel_cores = int(args[3])
        MoFGBMLBasicArgs.__train_file = args[4]
        MoFGBMLBasicArgs.__test_file = args[5]

    @staticmethod
    def get_train_file():
        return MoFGBMLBasicArgs.__train_file

    @staticmethod
    def get_test_file():
        return MoFGBMLBasicArgs.__test_file

    def __str__(self):
        # TODO
        return "not yet implemented"
