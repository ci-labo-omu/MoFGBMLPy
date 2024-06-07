from src.gbml.operator.crossover.uniform_crossover import UniformCrossover
from fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from gbml.solution.michigan_solution import MichiganSolution
from src.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from src.fuzzy.classifier.classifier import Classifier
from src.main.experience_parameter import ExperienceParameter
from src.data.input import Input
from data.output import Output
from src.main.consts import Consts
from src.main.basic.mofgbml_basic_args import MoFGBMLBasicArgs
from src.data.dataset_manager import DataSetManager
import sys
import os

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import random

from src.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from src.fuzzy.rule.rule_basic import RuleBasic
from fuzzy.knowledge.homo_triangle_knowledge_factory import HomoTriangleKnowledgeFactory


class MoFGBMLBasicMain:
    @staticmethod
    def main(args):
        # TODO: print information

        # Consts.set()
        #...

        Output.mkdirs(Consts.ROOTFOLDER)

        # set command arguments to static variables
        MoFGBMLBasicArgs.load(args)

        # Save const params
        file_name = str(os.path.join(Consts.EXPERIMENT_ID_DIR, "Consts.txt"))
        Output.writeln(file_name, str(Consts()), True)
        Output.writeln(file_name, str(MoFGBMLBasicArgs()), True)

        # Load dataset
        ExperienceParameter.get_instance().set_class_label_type(ExperienceParameter.ClassLabelType.SINGLE)
        Input.load_train_test_files(MoFGBMLBasicArgs.get_train_file(), MoFGBMLBasicArgs.get_test_file())
        manager = DataSetManager.get_instance()
        test = manager.get_tests()[0]
        train = manager.get_trains()[0]

        # Run the algo
        MoFGBMLBasicMain.hybrid_style_mofgbml(train, test)


    @staticmethod
    def hybrid_style_mofgbml(train, test):
        random.seed(2022)
        HomoTriangleKnowledgeFactory.create2_3_4_5(train.get_n_dim())

        bounds_Michigan = MichiganSolution.make_bounds()

        num_objectives_michigan = 1
        num_constraints_michigan = 0

        num_vars_pittsburgh = Consts.INITIATION_RULE_NUM
        num_objectives_pittsburgh = 2
        num_constraints_pittsburgh = 0

        rule_builder = RuleBasic.RuleBuilderBasic(HeuristicAntecedentFactory(train), LearningBasic(train))
        michigan_solution_builder = MichiganSolution.MichiganSolutionBuilder(bounds_Michigan,
                                                                             num_objectives_michigan,
                                                                             num_constraints_michigan,
                                                                             rule_builder)

        classification = SingleWinnerRuleSelection()
        classifier = Classifier(classification)

        problem = MoFGBMLBasicMain.BasicProblem(num_vars_pittsburgh,
                                                num_objectives_pittsburgh,
                                                num_constraints_pittsburgh,
                                                train,
                                                michigan_solution_builder,
                                                classifier)

        crossover_probability = 1

        algorithm = NSGA2(pop_size=Consts.POPULATION_SIZE,
                          sampling=MoFGBMLBasicMain.BasicSampling(train),
                          crossover=UniformCrossover(2, 1, crossover_probability),
                          mutation=MoFGBMLBasicMain.BasicMutation(train),
                          eliminate_duplicates=MoFGBMLBasicMain.BasicDuplicateElimination())

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 10),
                       seed=1,
                       verbose=True)

        cl = Classifier(SingleWinnerRuleSelection())
        res.X = [item[0] for item in res.X]
        non_dominated_solutions = res.pop

        print(res.X)
        for i in range(len(res.X)):
            print(res.X[i].get_class_label().get_class_label_value(), res.X[i].get_rule_weight().get_value())

        # print(cl.classify(res.X, train.get_patterns()[0]))


if __name__ == '__main__':
    MoFGBMLBasicMain.main(sys.argv[1:])
