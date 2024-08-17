import cython
from mofgbmlpy.main.arguments import Arguments
from mofgbmlpy.data.output import Output
import os


class MoFGBMLMOEADArgs(Arguments):
    """Arguments for MoFGBML for MOEAD"""
    def __init__(self):
        """Constructor"""
        super().__init__()

        args_definition = {
            "neighborhood-selection-probability": {
                "default": 1,
                "help": "Probability to select neighbors as parents",
                "type": "float",
                "required": False,
            },
            "neighborhood-size": {
                "default": 10,
                "help": "Number of neighbors, for each vector, considered per mating",
                "type": "int",
                "required": False,
            },
            "offspring-population-size": {
                "default": None,
                "help": "This argument has been kept for compatibility with the Java version jproperties file format, "
                        "but it's ignored",
                "type": "int",
                "required": False,
            }
        }

        for arg, arg_definition in args_definition.items():
            self._parser = Arguments.add_arg_to_parser(arg, arg_definition, self._parser)
