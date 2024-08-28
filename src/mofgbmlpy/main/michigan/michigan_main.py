from mofgbmlpy.main.abstract_main import AbstractMain


class MichiganMain(AbstractMain):
    def load_args(self, args, train=None, test=None):
        """Load the arguments

        Args:
            args (list): List of dash-case arguments
            train (Dataset): Training dataset
            test (Dataset): Test dataset
        """
        pass

    def run(self, args, train=None, test=None):
        """Main function of the runner

        Args:
            args (list): List of dash-case arguments
            train (Dataset): Training dataset
            test (Dataset): Test dataset

        Returns:
            pymoo.core.result.Result: Results of the run
        """
        pass
