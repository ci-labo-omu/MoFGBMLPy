import copy
import json
import xml.etree.cElementTree as xml_tree
import os

from jproperties import Properties

import argparse

from mofgbmlpy.main.arguments.arguments import Arguments


class PittsburghStyleArguments(Arguments):
    """Load and manage MoFGBML arguments for the Pittsburgh approach"""
    def __init__(self):
        super().__init__()
        self.load_config_file("pittsburgh_arguments")
