import cython
from mofgbmlpy.main.arguments import Arguments
from mofgbmlpy.data.output import Output
import os


class MoFGBMLNSGAIIIArgs(Arguments):
    """Arguments for MoFGBML for NSGA-III"""
    def __init__(self):
        """Constructor"""
        super().__init__()

        args_definition = {
            "offspring-population-size": {
                "default": None,
                "help": "Number of offsprings generated per generation",
                "type": "int",
                "required": False,
            }
        }

        for arg, arg_definition in args_definition.items():
            self._parser = Arguments.add_arg_to_parser(arg, arg_definition, self._parser)
