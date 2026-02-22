import xml.etree.cElementTree as xml_tree
import numpy as np
from matplotlib import pyplot as plt
from pymoo.core.callback import Callback
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation
from mofgbmlpy.gbml.operator.repair.pittsburgh_repair import PittsburghRepair
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder

from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from pymoo.core.population import Population
from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic

from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.arguments.pittsburgh_style_arguments import PittsburghStyleArguments
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
import sys


class PittsburghMain(AbstractMain):
    def __init__(self, knowledge_factory_class, algo_name):
        """Constructor

        Args:
            knowledge_factory_class (AbstractKnowledgeFactory): Knowledge factory class
        """
        args = PittsburghStyleArguments(algo_name)
        super().__init__(args, knowledge_factory_class)

    def _load_additional_args(self):
        self._callback = Callback()
        self._repair = PittsburghRepair()
        self._mutation = PittsburghMutation(self._knowledge, self._random_gen)
        self._sampling = HybridGBMLSampling(self._learner)

        pittsburgh_crossover = PittsburghCrossover(
            self._mofgbml_args.get("MIN_RULE_NUM"),
            self._mofgbml_args.get("MAX_RULE_NUM"),
            self._random_gen,
            self._mofgbml_args.get("PITTSBURGH_CROSS_RT"),
        )

        if self._mofgbml_args.get("CROSSOVER_TYPE") == "hybrid-gbml-crossover":
            crossover_probability = self._mofgbml_args.get("HYBRID_CROSS_RT")
            self._crossover = HybridGBMLCrossover(
                self._random_gen,
                self._mofgbml_args.get("MICHIGAN_OPE_RT"),
                MichiganCrossover(
                    self._mofgbml_args.get("RULE_CHANGE_RT"),
                    self._train,
                    self._knowledge,
                    self._mofgbml_args.get("MAX_RULE_NUM"),
                    self._random_gen,
                    self._mofgbml_args.get("MICHIGAN_CROSS_RT"),
                ),
                pittsburgh_crossover,
                crossover_probability,
            )
        elif self._mofgbml_args.get("CROSSOVER_TYPE") == "pittsburgh-crossover":
            self._crossover = pittsburgh_crossover
        else:
            raise ValueError("Unknown crossover type")

        num_objectives_michigan = 1
        num_constraints_michigan = 0

        num_vars_pittsburgh = self._mofgbml_args.get("INITIATION_RULE_NUM")
        num_constraints_pittsburgh = 0

        michigan_solution_builder = MichiganSolutionBuilder(
            self._random_gen, num_objectives_michigan, num_constraints_michigan, self._rule_builder
        )

        # classification = SingleWinnerRuleSelection(self._mofgbml_args.get("CACHE_SIZE"))
        classification = SingleWinnerRuleSelection()

        self._problem = PittsburghProblem(
            num_vars_pittsburgh,
            self._objectives,
            num_constraints_pittsburgh,
            self._train,
            michigan_solution_builder,
            classification,
        )

    @staticmethod
    def plot_line_interpretability_error_rate_tradeoff(
        solutions, file_path=None, title=None, xlim=None, grid=True, x_key="total_rule_length"
    ):
        """Plot an interpretability error rate tradeoff of the solutions

        Args:
            solutions (PittsburghSolution[]): solutions
            file_path (str): Path of the file where the plot will be saved
            title (str): Title of the plot
            xlim (tuple): X-axis domain shown
            grid (bool): If true then show a grid
            x_key (str): Key of the value in the dict used as the X-axis
        """
        err_train = []
        err_test = []

        if x_key == "total_rule_length":
            x_label = "Total rule length"
        elif x_key == "num_rules":
            x_label = "Num rules"
        else:
            raise ValueError("only total_rule_length and num_rules are accepted for the x_key")

        for solution in solutions:
            err_train.append((solution.get_attribute(x_key), solution.get_attribute("training_error_rate")))
            err_test.append((solution.get_attribute(x_key), solution.get_attribute("test_error_rate")))

        PittsburghMain.plot_line_interpretability_error_rate_tradeoff_from_coords(
            err_train,
            err_test,
            x_label=x_label,
            y_label="Error rate",
            file_path=file_path,
            title=title,
            xlim=xlim,
            grid=grid,
        )

    @staticmethod
    def plot_line_interpretability_error_rate_tradeoff_from_coords(
        err_train,
        err_test,
        x_label="Total rule length",
        y_label="Error rate",
        file_path=None,
        title=None,
        xlim=None,
        grid=True,
    ):
        """Plot an interpretability error rate tradeoff from coordinates

        Args:
            err_train (list): List of tuples (x_value, err_train_value_at_x)
            err_test (list): List of tuples (x_value, err_test_value_at_x)
            x_label (str): Name of the X-axis label
            y_label (str): Name of the Y-axis label
            file_path (str): Path of the file where the plot will be saved
            title (str): Title of the plot
            xlim (tuple): X-axis domain shown
            grid (bool): If true then show a grid
        """
        err_train = list(set(err_train))
        err_train.sort()
        for i in range(len(err_train)):
            err_train[i] = list(err_train[i])  # tuple to list
        err_train = np.array(err_train)

        err_test = list(set(err_test))
        err_test.sort()
        for i in range(len(err_test)):
            err_test[i] = list(err_test[i])  # tuple to list
        err_test = np.array(err_test)

        if len(err_train) != 0:
            plt.plot(err_train[:, 0], err_train[:, 1], c="darkorange", marker="o", label="Train")
        if len(err_test) != 0:
            plt.plot(err_test[:, 0], err_test[:, 1], c="blue", marker="o", label="Test")
        plt.xlabel(x_label)
        if title is not None:
            plt.title(title)
        if grid:
            plt.grid()

        plt.ylabel(y_label)
        plt.ylim(0, 1)
        if xlim is not None:
            plt.xlim(xlim)

        if len(err_train) != 0 or len(err_test) != 0:
            plt.legend(loc="upper left")

        if file_path is not None:
            plt.savefig(file_path)

        plt.show()

    @staticmethod
    def update_results_data(solutions, knowledge, train, test, id_start=0):
        """Update the solutions data (attributes)

        Args:
            solutions (PittsburghSolution[]): solutions
            knowledge (Knowledge): Knowledge base
            train (Dataset): Training dataset
            test (Dataset):  Test dataset
            id_start (int): The ID is determined by the order of the solutions in loop.
                This parameter determines the starting value for the ID
        """
        if id_start < 0:
            raise ValueError("ID must be positive or null")

        sol_id = id_start
        for i in range(len(solutions)):
            total_coverage = 0
            sol = solutions[i]
            for rule_i in range(sol.get_num_vars()):
                michigan_solution = sol.get_var(rule_i)
                fuzzy_set_indices = michigan_solution.get_vars()
                coverage = 1
                for dim_i in range(len(fuzzy_set_indices)):
                    coverage *= knowledge.get_support(dim_i, fuzzy_set_indices[dim_i])
                total_coverage += coverage

            sol.set_attribute("id", sol_id)
            sol.set_attribute("total_coverage", total_coverage)
            sol.set_attribute("total_rule_length", sol.get_total_rule_length())
            sol.set_attribute("average_rule_weight", sol.get_average_rule_weight())
            sol.set_attribute("training_error_rate", sol.calc_error_rate(train))
            sol.set_attribute("test_error_rate", sol.calc_error_rate(test))
            sol.set_attribute("num_rules", sol.get_num_vars())

            sol_id += 1

    @staticmethod
    def import_xml_classifiers(
        file_path, train_file_path=None, test_file_path=None, is_multi_label=False, objectives=None
    ):
        """Import classifiers from an XML file

        Args:
            file_path (str): Path of the XML file
            train_file_path (str): Path of the training data file
            test_file_path (str): Path of the test data file
            is_multi_label (bool): If true then the dataset is multi-label
            objectives (Objective[]): Objectives used in the problem

        Returns:
            PittsburghSolution[]: List of Pittsburgh solutions
            Knowledge: Knowledge base
            Arguments: MoFGBML arguments
        """
        tree = xml_tree.parse(file_path)
        root = tree.getroot()

        consts_xml = root.find("consts")

        args = PittsburghStyleArguments.from_xml(consts_xml)
        knowledge = None
        classifiers = []

        generation_xml = root.findall("generations")[-1]

        knowledge_xml = generation_xml.find("knowledgeBase")

        if knowledge_xml is not None:
            knowledge = Knowledge.from_xml(knowledge_xml)

        if not args.has_key("TRAIN_FILE"):
            if train_file_path is None:
                raise ValueError("Train file path must be provided")
            args.set("TRAIN_FILE", train_file_path)
        if not args.has_key("TEST_FILE"):
            if test_file_path is None:
                test_file_path = train_file_path
            args.set("TEST_FILE", test_file_path)
        if not args.has_key("IS_MULTI_LABEL"):
            args.set("IS_MULTI_LABEL", is_multi_label)
        if not args.has_key("OBJECTIVES"):
            if objectives is None:
                raise ValueError("Objectives must be provided")
            args.set("OBJECTIVES", objectives)

        training_data_set, _ = Input.get_train_test_files(args)
        is_dc_probability = args.get("IS_PROBABILITY_DONT_CARE")
        dc_rate = args.get("DONT_CARE_RT")
        antecedent_number_do_not_dont_care = args.get("ANTECEDENT_NUMBER_DO_NOT_DONT_CARE")
        num_objectives = len(args.get("OBJECTIVES"))
        num_constraints = 0
        num_vars = args.get("INITIATION_RULE_NUM")

        population_xml = generation_xml.find("population")

        random_gen = np.random.Generator(np.random.MT19937(seed=2022))
        antecedent_factory = HeuristicAntecedentFactory(
            training_data_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen
        )
        consequent_factory = LearningBasic(training_data_set)

        objectives = PittsburghMain._get_objectives_static(args, training_data_set, True)
        classification = SingleWinnerRuleSelection()

        rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)
        michigan_solution_builder = MichiganSolutionBuilder(random_gen, 2, 0, rule_builder)

        problem = PittsburghProblem(
            num_vars, objectives, num_constraints, training_data_set, michigan_solution_builder, classification
        )

        if population_xml is not None:
            for classifier_xml in population_xml.findall("pittsburghSolution"):
                classifier = PittsburghSolution.from_xml(
                    classifier_xml,
                    random_gen,
                    knowledge,
                    num_objectives,
                    num_constraints,
                    rule_builder,
                    classification,
                    michigan_solution_builder,
                )

                classifiers.append([classifier])

        classifiers = Population.new(X=np.array(classifiers))

        problem.evaluate(classifiers.get("X"))

        return classifiers, knowledge, args


if __name__ == "__main__":
    algo_name = AbstractMain.get_algo_name_from_raw_args(sys.argv[1:])
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
    runner.run(sys.argv[1:])
