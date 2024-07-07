import xml.etree.cElementTree as xml_tree
import os
from abc import ABC, abstractmethod

from mofgbmlpy.data.output import Output
import argparse

class Arguments(ABC):
    __values = None
    __required_args = None
    __bool_args = None

    def __init__(self):
        self.__values = {}
        self.__required_args = [
            "data-name",
            "algorithm-id",
            "experiment-id",
            "num-parallel-cores",
            "train-file",
            "test-file"
        ]
        self.__bool_args = [
            "is-dont-care-probability",
            "is-multi-label",
            "no-plot",
            "pretty-xml"
        ]

        # TODO: Check if those parameters are all useful

        # Parallelization
        self.__values["NUM_PARALLEL_CORES"] = None


        # Experimental Settings
        self.__values["EXPERIMENT_ID"] = None
        self.__values["ALGORITHM_ID"] = None
        self.__values["POPULATION_SIZE"] = 60
        self.__values["OFFSPRING_POPULATION_SIZE"] = 60
        self.__values["TERMINATE_GENERATION"] = 500
        self.__values["TERMINATE_EVALUATION"] = 5000 #300000
        # self.__values["OUTPUT_FREQUENCY"] = 6000

        # Random Number Seed
        self.__values["RAND_SEED"] = 2020

        # OS
        # self.__values["WINDOWS"] = 0     # Windows
        # self.__values["UNIX"] = 1        # Mac or Linux

        # Fuzzy Classifier
        self.__values["IS_DONT_CARE_PROBABILITY"] = False
        self.__values["ANTECEDENT_NUM_NOT_DONT_CARE"] = 5
        self.__values["DONT_CARE_RT"] = 0.8
        self.__values["INITIATION_RULE_NUM"] = 30
        self.__values["MAX_NUM_RULES"] = 60
        self.__values["MIN_NUM_RULES"] = 1

        # FGBML
        self.__values["HYBRID_CROSS_RT"] = 1
        self.__values["MICHIGAN_OPE_RT"] = 0.5
        self.__values["RULE_CHANGE_RT"] = 0.2
        self.__values["MICHIGAN_CROSS_RT"] = 0.9
        self.__values["PITTSBURGH_CROSS_RT"] = 0.9
        self.__values["FUZZY_GRADE"] = 1.0

        # Folders' Name
        self.__values["ROOT_FOLDER"] = "results"
        self.__values["ALGORITHM_ID_DIR"] = "ALGORITHM_ID"
        self.__values["EXPERIMENT_ID_DIR"] = "EXPERIMENT_ID"

        # Index
        # self.__values["TRAIN"] = 0
        # self.__values["TEST"] = 1
        # self.__values["XML_FILE_NAME"] = "results_XML"

        # Dataset info
        self.__values["DATA_SIZE"] = 0
        self.__values["ATTRIBUTE_NUMBER"] = 0
        # self.__values["CLASS_LABEL_NUMBER"] = 0
        self.__values["IS_MULTI_LABEL"] = False
        self.__values["TRAIN_FILE"] = None
        self.__values["TEST_FILE"] = None
        self.__values["DATA_NAME"] = None

        # Results and display
        self.__values["NO_PLOT"] = False
        self.__values["PRETTY_XML"] = False

    def set(self, key, value):
        self.__values[str(key)] = value

    def get(self, key):
        return self.__values[key]

    def get_keys(self):
        return list(self.__values.keys())

    def has_key(self, key):
        return key in self.__values

    def __str__(self):
        txt = ""
        for key, value in self.__values.items():
            txt += f"{key} = {value}\n"

        return txt

    def parse_args(self, args):
        parser = argparse.ArgumentParser()
        args_dict_keys = [Arguments.key_to_arg(key) for key in self.__values.keys()]

        for arg in args_dict_keys:
            if arg in self.__required_args:
                is_required = True
            else:
                is_required = False

            if arg in self.__bool_args:
                parser.add_argument("--" + arg, required=is_required, action="store_true"),
            else:
                parser.add_argument("--"+arg, required=is_required)


        # Remove not specified args and return a dict of the args
        returned_args = {}
        for arg, value in vars(parser.parse_args(args)).items():
            if value is not None:
                returned_args[Arguments.arg_to_key(arg)] = value

        return returned_args

    @staticmethod
    def arg_to_key(arg):
        return arg.upper().replace("-", "_")

    @staticmethod
    def key_to_arg(arg):
        return arg.lower().replace("_", "-")

    def load(self, args):
        parsed_args = self.parse_args(args)

        for key, val in parsed_args.items():
            self.set(key, val)

        self.set("ALGORITHM_ID_DIR", str(os.path.join(self.get("ROOT_FOLDER"), str(self.get("ALGORITHM_ID")))))

        Output.mkdirs(self.get("ALGORITHM_ID_DIR"))


        self.set("EXPERIMENT_ID_DIR",
                 str(os.path.join(
                     self.get("ALGORITHM_ID_DIR"),
                     self.get("DATA_NAME"),
                     str(self.get("EXPERIMENT_ID")))))
        Output.mkdirs(self.get("EXPERIMENT_ID_DIR"))

    def to_xml(self):
        root = xml_tree.Element("consts")
        for key, value in self.__values.items():
            term_xml = xml_tree.SubElement(root, key)
            term_xml.text = str(value)

        return root