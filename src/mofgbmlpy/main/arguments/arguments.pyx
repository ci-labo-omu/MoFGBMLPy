import copy
import json
import xml.etree.cElementTree as xml_tree
import os

from jproperties import Properties

import argparse


class Arguments:
    """Load and manage MoFGBML arguments

    Attributes:
        __args_definition (object): Arguments definitions (e.g. specifies the name, default value, ...)
        __values (dict): Values for each argument (initialized after calling the load function)
        _parser (argparse.ArgumentParser): Arguments parser (check if there are missing arguments, if the type is valid...)
    """
    def __init__(self):
        """Constructor"""
        # TODO: change antecedent factory and other args via these params

        self.__args_definition = {}
        self.__values = {}
        self._parser = argparse.ArgumentParser()
        self.__config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        self.load_config_file("base_arguments")

    def add_args_dict_to_parser(self, args_dict):
        """Load arguments dictionary in the parser

        Args:
            args_dict (dict): Arguments dictionary
        """
        for arg, arg_definition in args_dict.items():
            self._parser = Arguments.add_arg_to_parser(arg, arg_definition, self._parser)


    def load_config_file(self, data_file_name):
        """Load a JSON file from a file name (NOT the path) and load its arguments in the parser

        Args:
            data_file_name (str): Name of a JSON file without extension
        """

        path = self.__config_root + "/" + data_file_name + ".json"
        with open(path) as f:
            data = json.load(f)
            for k, v in data.items():
                if k in self.__args_definition:
                    raise Exception(f"Duplicated arguments are not yet handled. {str(k)} is already loaded")  # TODO
                else:
                    self.__args_definition[k] = v

    def load_parser(self):
        """Load the arguments definition into the parser"""
        self.add_args_dict_to_parser(self.__args_definition)

    def set(self, key, value):
        """Set the value of an argument

        Args:
            key (str): Argument name
            value (object): New value
        """
        self.__values[str(key)] = value

    def get(self, key):
        """Get the value of an argument

        Args:
            key (str): Name of the argument to be fetched

        Returns:
            object: Fetched argument
        """
        return self.__values[key]

    def get_keys(self):
        """Get the list of keys (argument names)

        Returns:
            list: List of keys
        """
        return list(self.__values.keys())

    def has_key(self, key):
        """Check if an argument exists after load

        Args:
            key (str): Key of the argument checked

        Returns:
            bool: True if it exists and false otherwise
        """
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
        """Add an argument to the parser

        Args:
            arg (str): Argument name in dash case
            arg_definition (dict): Argument definition properties (default value, ...)
            parser (argparse.ArgumentParser) Parser to which arguments are added

        Returns:
            parser (argparse.ArgumentParser) Parser with the added arguments
        """
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
        """Parse a list of arguments

        Args:
            args (list): List of arguments in dash-case, e.g. ["--verbose", "--dont-care-rt", "0.5"]

        Returns:
            dict: Parsed args as a dictionary with keys in capital letters and snake case
        """
        # Remove not specified args and return a dict of the args
        returned_args = {}
        for arg, value in vars(self._parser.parse_args(args)).items():
            returned_args[Arguments.arg_to_key(arg)] = value

        return returned_args

    def get_args_from_jproperties(self):
        """Get arguments list (dash-case) from a consts.properties file

        Returns:
            list: List of arguments in dash-case
        """
        root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
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

    @staticmethod
    def arg_to_key(arg):
        """Convert a text in the format of an argument to one of a key, e.g. "verbose" becomes "VERBOSE"

        Args:
            arg (str): Text to be converted

        Returns:
            str: Formatted text
        """
        return arg.upper().replace("-", "_")

    @staticmethod
    def key_to_arg(arg):
        """Convert a text in the format of a key to one of an argument, e.g. "VERBOSE" becomes "verbose"

        Args:
            arg (str): Text to be converted

        Returns:
            str: Formatted text
        """
        return arg.lower().replace("_", "-")

    def dict_to_list(self, args_dict):
        args_list = []
        for k, v in args_dict.items():
            if type(v) == type(True) and v:
                args_list.append("--"+k)

    def get_accepted_arguments(self):
        accepted_args = []
        for k in self.__args_definition.keys():
            if k is "exclusive-groups":
                for group in self.__args_definition["exclusive-groups"]:
                    for k2 in group.keys():
                        accepted_args.append(k2)
            else:
                accepted_args.append(k2)

        return accepted_args

    def load(self, args):
        """Load arguments into this object. the file consts.properties in the root folder can also be used, but the priority will be give nto the values in the args parameter

        Args:
            args (list): List of arguments in dash-case
        """
        new_args = copy.deepcopy(args)
        args_from_jproperties = self.get_args_from_jproperties()

        # Add arg in args_from_properties to args if they are not already present
        i = 0
        while i < len(args_from_jproperties):
            do_skip = False
            if args_from_jproperties[i] not in new_args:

                # Check if this argument can't be used simultaneously with another one
                for exclusive_group in self.__args_definition["exclusive-groups"]:
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

        self.set("ALGORITHM_ID_DIR", str(os.path.join(str(self.get("ROOT_FOLDER")), str(self.get("ALGORITHM_ID")))))


        self.set("EXPERIMENT_ID_DIR",
                 str(os.path.join(
                     str(self.get("ALGORITHM_ID_DIR")),
                     str(self.get("DATA_NAME")),
                     str(self.get("EXPERIMENT_ID")))))


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