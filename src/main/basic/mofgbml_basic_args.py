from src.main.consts import Consts
import os

class MoFGBMLBasicArgs:
    __data_name = None
    __algorithm_id = None
    __experiment_id = None
    __num_parallel_cores = None
    __train_file = None
    __test_file = None

    def load(self, args):
        if len(args) < 6:
            raise Exception("Not enough arguments (6 were expected)")

        self.__data_name = args[0]

        self.__algorithm_id = args[1]
        Consts.ALGORITHM_ID_DIR = str(os.path.join(Consts.ROOTFOLDER, self.__algorithm_id))
        Output.mkdirs(Consts.ALGORITHM_ID_DIR)

        self.__experiment_id = args[2]
        # todo: change "/" to join()
        Consts.EXPERIMENT_ID_DIR = Consts.ALGORITHM_ID_DIR + "/" + self.__data_name + "/" + self.__experiment_id
        Output.mkdirs(Consts.EXPERIMENT_ID_DIR)

        self.__num_parallel_cores = int(args[3])
        self.__train_file = args[4]
        self.__test_file = args[5]
