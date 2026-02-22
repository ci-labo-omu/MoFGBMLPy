from mofgbmlpy.main.abstract_main import AbstractMain


class MichiganMain(AbstractMain):
    """MoFGBML runner for Michigan-style individuals."""

    def load_args(self, args, train=None, test=None):
        """Load the arguments.

        Args:
            args (list): List of dash-case arguments
            train (Dataset): Training dataset
            test (Dataset): Test dataset
        """
        pass
