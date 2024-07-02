import cython
from mofgbmlpy.main.arguments import Arguments
from mofgbmlpy.data.output import Output
import os


class MoFGBMLMOEADArgs(Arguments):
    def _load(self, args):
        self.set("LEARNING_EXPERIMENT_ID_DIR", "EXPERIMENT_ID")
        self.set("NEIGHBORHOOD_SELECTION_PROBABILITY", 1)
        self.set("MAXIMUM_NUMBER_OF_REPLACED_SOLUTIONS", 10)
        self.set("NEIGHBORHOOD_SIZE", 10)
