import cython
from mofgbmlpy.main.arguments import Arguments
from mofgbmlpy.data.output import Output
import os


class MoFGBMLMOEADArgs(Arguments):
    def __init__(self):
        super().__init__()
        self.set("LEARNING_EXPERIMENT_ID_DIR", "EXPERIMENT_ID")
        self.set("NEIGHBORHOOD_SELECTION_PROBABILITY", 1)
        self.set("MAXIMUM_NUMBER_OF_REPLACED_SOLUTIONS", 10)
        self.set("NEIGHBORHOOD_SIZE", 10)

        self._int_args.append("neighbour-selection-probability")
        self._int_args.append("maximum-number-of-replaced-solutions")
        self._int_args.append("neighborhood-size")
