from src.main.experience_parameter import ExperienceParameter
from src.data.input import Input

class MofgbmlBasicMain:
    def main(args):
        # TODO: print information

        # Consts.set()
        #...

        # Load dataset
        params = ExperienceParameter.get_instance()
        params.set_class_label_type(ExperienceParameter.ClassLabelType.SINGLE)

        Input.load_train_test_files(MoFGBML_MOEAD_CommandLineArgs.trainFile, MoFGBML_MOEAD_CommandLineArgs.testFile)

