import xml.etree.cElementTree as xml_tree
import os
from abc import ABC, abstractmethod

from mofgbmlpy.data.output import Output
import argparse

class Arguments(ABC):
    __values = None
    _parser = None

    def __init__(self):
        # TODO: change antecedent factory and other args via these params

        args_definition = {
            # Parallelization
            # "num-parallel-cores": {
            #     "default": 1,
            #     "help": "Number of cores used for parallelization",
            #     "type": "int",
            #     "required": False
            # },

            # Experimental Settings
            "experiment-id": {
                "default": None,
                "help": "Experiment ID (Used to create the path where results are saved)",
                "type": "string",
                "required": True
            },
            "algorithm-id": {
                "default": None,
                "help": "Algorithm ID (Used to create the path where results are saved)",
                "type": "string",
                "required": True
            },
            "population-size": {
                "default": 60,
                "help": "Population size (Number of individuals per generation)",
                "type": "int",
                "required": False,
            },
            "exclusive-groups": [
                {
                    "terminate-generation": {
                        "default": None,
                        "help": "Set the termination criterion to the number of generation, and set the max value",
                        "type": "int",
                        "required": False,
                    },
                    "terminate-evaluation": {
                        "default": None,
                        "help": "Set the termination criterion to the number of objective function evaluations, and set the max value",
                        "type": "int",
                        "required": False,
                    }
                }
            ],
            # "output-frequency": { # TODO
            #     "default": 5000,
            #     "help": "",
            #     "type": "int",
            #     "required": False,
            # },

            # Random Number Seed
            "rand-seed": {
                "default": 2020,
                "help": "The seed for random operations",
                "type": "int",
                "required": False,
            },

            # Fuzzy Classifier
            "is-dont-care-probability": {
                "default": True,
                "help": "If True then use the don't care rate for the antecedent factory, otherwise compute it from antecedent num not don't care",
                "type": "bool",
                "required": False,
            },
            "antecedent-num-not-dont-care": {
                "default": 5,
                "help": "Number of indices that are not 0 (which is don't care) in an antecedent. Used by the antecedent factory",
                "type": "int",
                "required": False,
            },
            "dont-care-rt": {
                "default": 0.8,
                "help": "Don't care probability for antecedent indices in antecedent factory",
                "type": "float",
                "required": False,
            },
            "initiation-rule-num": {
                "default": 30,
                "help": "Number of rules in Pittsburgh solutions in the initial population",
                "type": "int",
                "required": False,
            },
            "max-num-rules": {
                "default": 60,
                "help": "Maximum number of rules in Pittsburgh solutions",
                "type": "int",
                "required": False,
            },
            "min-num-rules": {
                # Check if it doesn't cause issues when eliminating invalid rules
                "default": 1,
                "help": "Minimum number of rules in Pittsburgh solutions",
                "type": "int",
                "required": False,
            },
            "antecedent-factory": {
                "default": "heuristic-antecedent-factory",
                "help": "Antecedent factory used for fuzzy rule generation. If crossover-type is hybrid then",
                "choices": ["all-combination-antecedent-factory", "heuristic-antecedent-factory"],
                "type": "string",
                "required": False
            },
            "crossover-type": {
                "default": "hybrid-gbml-crossover",
                "help": "Crossover used in the GA algorithm",
                "choices": ["hybrid-gbml-crossover", "pittsburgh-crossover"],
                "type": "string",
                "required": False
            },

            # FGBML
            "hybrid-cross-rt": {
                "default": 1,
                "help": "Probability that a (hybrid) crossover occurs",
                "type": "float",
                "required": False,
            },
            "michigan-ope-rt": {
                "default": 0.5,
                "help": "Probability that Michigan mating operators are used instead of a Pittsburgh one",
                "type": "float",
                "required": False,
            },
            "rule-change-rt": {
                "default": 0.2,
                "help": "Ratio of the rules that are changed in a Michigan crossover",
                "type": "float",
                "required": False,
            },
            "michigan-cross-rt": {
                "default": 0.9,
                "help": "Probability that a Michigan crossover occurs",
                "type": "float",
                "required": False,
            },
            "pittsburgh-cross-rt": {
                "default": 0.9,
                "help": "Probability that a Pittsburgh crossover occurs",
                "type": "float",
                "required": False,
            },
            # "FUZZY-GRADE": { # TODO: not yet implemented
            #     "default": 1.0,
            #     "help": "",
            #     "type": "float",
            #     "required": False,
            # },
            "objectives": {
                "default": [],
                "help": "List of the objectives. Accepted values: 'error-rate', 'rule-interpretation', 'num-rules', 'total-rule-length",
                "type": "list",
                "required": True,
            },

            # Folders' Name
            "root-folder": {
                "default": "results",
                "help": "Path where results are saved",
                "type": "string",
                "required": False,
            },

            # Dataset info
            "is-multi-label": {
                "default": False,
                "help": "Must be True if the dataset is a multi label one and False otherwise, which is the default",
                "type": "bool",
                "required": False,
            },
            "train-file": {
                "default": None,
                "help": "Path of the training dataset file",
                "type": "string",
                "required": True,
            },
            "test-file": {
                "default": None,
                "help": "Path of the test dataset file",
                "type": "string",
                "required": True,
            },
            "data-name": {
                "default": None,
                "help": "Dataset name. It's used to create the path where results are saved",
                "type": "string",
                "required": True,
            },

            # Results and display
            "no-plot": {
                "default": True,
                "help": "Don't generate matplotlib plots",
                "type": "bool",
                "required": False,
            },
            "pretty-xml": {
                "default": False,
                "help": "Save results in a pretty XML file (formated with indentation)",
                "type": "bool",
                "required": False,
            },
        }

        self.__values = {}
        self._parser = argparse.ArgumentParser()

        for arg, arg_definition in args_definition.items():
            self._parser = Arguments.add_arg_to_parser(arg, arg_definition, self._parser)


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

    @staticmethod
    def add_arg_to_parser(arg, arg_definition, parser):
        if arg == "exclusive-groups":
            for group_data in arg_definition:
                group = parser.add_mutually_exclusive_group(required=True)
                for arg_in_group, arg_definition_in_group in group_data.items():
                    group = Arguments.add_arg_to_parser(arg_in_group, arg_definition_in_group, group)
        elif arg_definition["type"] == "bool":
            parser.add_argument("--" + arg, required=arg_definition["required"], action="store_true", default=arg_definition["default"], help=arg_definition["help"]),
        elif arg_definition["type"] == "float":
            parser.add_argument("--" + arg, required=arg_definition["required"], type=float, default=arg_definition["default"], help=arg_definition["help"])
        elif arg_definition["type"] == "int":
            parser.add_argument("--" + arg, required=arg_definition["required"], type=int, default=arg_definition["default"], help=arg_definition["help"])
        elif arg_definition["type"] == "list":
            parser.add_argument("--" + arg, nargs='+', required=arg_definition["required"], help=arg_definition["help"])
        else:
            parser.add_argument("--" + arg, required=arg_definition["required"], default=arg_definition["default"], help=arg_definition["help"])

        return parser

    def parse_args(self, args):
        # Remove not specified args and return a dict of the args
        returned_args = {}
        for arg, value in vars(self._parser.parse_args(args)).items():
            returned_args[Arguments.arg_to_key(arg)] = value

        return returned_args

    @staticmethod
    def arg_to_key(arg):
        return arg.upper().replace("-", "_")

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