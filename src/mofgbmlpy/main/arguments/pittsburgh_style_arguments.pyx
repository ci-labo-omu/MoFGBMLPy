import copy
import xml.etree.cElementTree as xml_tree
import os

from jproperties import Properties

import argparse

from mofgbmlpy.main.arguments.arguments import Arguments


class PittsburghStyleArguments(Arguments):
    """Load and manage MoFGBML arguments for the Pittsburgh approach"""
    def __init__(self, algo_name):
        super().__init__()
        self.load_config_file("pittsburgh_arguments")
        self.load_config_file(algo_name+"_arguments")
        self.load_parser()
        self.set("IS_MICHIGAN_STYLE", False)

    @staticmethod
    def from_xml(xml_element):
        """Load an Arguments object from an XML element

        Args:
            xml_element (xml.etree.ElementTree): XML element representing an Arguments object

        Returns:
            (PittsburghStyleArguments): Loaded Arguments object
        """

        algo_xml = xml_element.find("ALGORITHM")
        algo_name = algo_xml.text if algo_xml is not None else "nsga2"
        args = PittsburghStyleArguments(algo_name)

        for child in xml_element:
            try:
                val_type = args.get_type(child.tag)

                if val_type is not None:
                    if val_type == "string":
                        args.set(child.tag, child.text)
                    elif val_type == "bool":
                        args.set(child.tag, child.text.lower() == "true")
                    elif val_type == "list":
                        list_str = child.text.strip()[1:-1]
                        list_items = [item.strip()[1:-1] for item in list_str.split(",")]
                        args.set(child.tag, list_items)
                    else:
                        val_type = eval(val_type)
                        args.set(child.tag, val_type(child.text))
            except Exception:
                print(f"Couldn't load argument: {child.tag} from XML")
        return args
