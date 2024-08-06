import copy
import xml.etree.cElementTree as xml_tree
import os
from abc import ABC, abstractmethod

import numpy as np
from jproperties import Properties

from mofgbmlpy.data.output import Output
import argparse

class Arguments(ABC):
    __exclusive_groups = None
    __values = None
    _parser = None

    def __init__(self):
        # TODO: change antecedent factory and other args via these params

        args_definition = {
            # Optimization
            # "num-parallel-cores": {
            #     "default": 1,
            #     "help": "Number of cores used for parallelization",
            #     "type": "int",
            #     "required": False
            # },
            "cache-size": {
                "default": 0,
                "help": "Third argument has been left for test purposes but it's not recommended to use it since the hashing function has collisions. Cache size for fitness values computation. A big cache might deteriorate performance and increase RAM usage",
                "type": "int",
                "required": False
            },

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
            "verbose": {
                "default": False,
                "help": "If True then display for each generation some information",
                "type": "bool",
                "required": False,
            },

            # Random Number Seed
            "rand-seed": {
                "default": 2020,
                "help": "The seed for random operations",
                "type": "int",
                "required": False,
            },

            # Fuzzy Classifier
            "is-probability-dont-care": {
                "default": False,
                "help": "If specified then use the don't care rate for the antecedent factory, otherwise compute it from antecedent num not don't care",
                "type": "bool",
                "required": False,
            },
            "antecedent-number-do-not-dont-care": {
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
            # "FUZZY-GRADE": { # TODO: not yet implemented. It seems to be used for entropy division, which is used by trapezoidal fuzzy sets
            #     "default": 1.0,
            #     "help": "",
            #     "type": "float",
            #     "required": False,
            # },
            "objectives": {
                "default": ["num-rules", "error-rate"],
                "help": "List of the objectives. Accepted values: 'error-rate', 'rule-interpretation', 'num-rules', 'total-rule-length",
                "type": "list",
                "required": False,
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
                "help": "Must be specified if the dataset is a multi label one and not specified otherwise, which is the default",
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
            "gen-plot": {
                "default": False,
                "help": "Generate matplotlib plots",
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
        self.__exclusive_groups = args_definition["exclusive-groups"]
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

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
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
            parser.add_argument("--" + arg, nargs='+', required=arg_definition["required"], default=arg_definition["default"], help=arg_definition["help"])
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

    @staticmethod
    def key_to_arg(arg):
        return arg.lower().replace("_", "-")

    def get_args_from_jproperties(self):
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        file_path = root_folder + '/consts.properties'

        if not os.path.exists(file_path):
            return []

        configs = Properties()
        with open(file_path, 'rb') as read_prop:
            configs.load(read_prop)

        prop_view = configs.items()

        args = []

        #--max-rule-num 60 --min-rule-num 1

        for item in prop_view:
            key = Arguments.key_to_arg(item[0])
            value = str(item[1].data)
            
            # Translate Java version args format to this version format
            if key == "antecedent-len":
                key = "antecedent-number-do-not-dont-care"
            elif key == "max-rule-num":
                key = "max-num-rules"
            elif key == "min-rule-num":
                key = "min-num-rules"

            if value == "true":
                args = args + ["--"+key]
            elif value == "false":
                continue
            else:
                args = args + ["--"+key, value]

        return args

    def load(self, args):
        new_args = copy.deepcopy(args)
        args_from_jproperties = self.get_args_from_jproperties()

        # Add arg in args_from_properties to args if they are not already present
        i = 0
        while i < len(args_from_jproperties):
            do_skip = False
            if args_from_jproperties[i] not in new_args:

                # Check if this argument can't be used simultaneously with another one
                for exclusive_group in self.__exclusive_groups:
                    keys = [f"--{k}" for k in exclusive_group.keys()]
                    if args_from_jproperties[i] in keys:
                        for key in keys:
                            if key in new_args:
                                # Already in the given args, so we can't add it since it's mutually exclusive
                                do_skip = True
                                break
                        break
            else:
                do_skip = True

            if do_skip:
                i += 1
                while i < len(args_from_jproperties) and args_from_jproperties[i][0] != "-":
                    i += 1
                    continue
            else:
                new_args.append(args_from_jproperties[i])
                i += 1
                while i < len(args_from_jproperties) and args_from_jproperties[i][0] != "-":
                    new_args.append(args_from_jproperties[i])
                    i += 1

        parsed_args = self.parse_args(new_args)

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
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("consts")
        for key, value in self.__values.items():
            term_xml = xml_tree.SubElement(root, key)
            term_xml.text = str(value)

        return root