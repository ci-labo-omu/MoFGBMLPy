from importlib import import_module

import numpy as np
from matplotlib import pyplot as plt
from pymoo.core.callback import Callback
from pymoo.termination import get_termination

from mofgbmlpy.data.input import Input
from mofgbmlpy.data.output import Output
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_multi import LearningMulti
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.fuzzy.rule.rule_builder_multi import RuleBuilderMulti
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring_michigan import \
    UniformCrossoverSingleOffspringMichigan
from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation
from mofgbmlpy.gbml.operator.repair.pittsburgh_repair import PittsburghRepair
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.restore_population_if_worse_michigan import RestorePopulationIfWorseMichigan
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.arguments.arguments import Arguments
from mofgbmlpy.main.arguments.pittsburgh_style_arguments import PittsburghStyleArguments
from mofgbmlpy.utility.util import dash_case_to_class_name, dash_case_to_snake_case


class PittsburghMain(AbstractMain):
    def __init__(self, knowledge_factory_class):
        """Constructor

        Args:
            knowledge_factory_class (AbstractKnowledgeFactory): Knowledge factory class
        """
        args = PittsburghStyleArguments()
        super().__init__(args, knowledge_factory_class)

    def _load_additional_args(self):
        self._callback = Callback()
        self._repair = PittsburghRepair()
        self._mutation = PittsburghMutation(self._knowledge, self._random_gen)
        self._sampling = HybridGBMLSampling(self._learner)

        pittsburgh_crossover = PittsburghCrossover(self._mofgbml_args.get("MIN_NUM_RULES"),
                                                   self._mofgbml_args.get("MAX_NUM_RULES"),
                                                   self._random_gen,
                                                   self._mofgbml_args.get("PITTSBURGH_CROSS_RT"))

        if self._mofgbml_args.get("CROSSOVER_TYPE") == "hybrid-gbml-crossover":
            crossover_probability = self._mofgbml_args.get("HYBRID_CROSS_RT")
            self._crossover = HybridGBMLCrossover(self._random_gen,
                                                  self._mofgbml_args.get("MICHIGAN_OPE_RT"),
                                                  MichiganCrossover(
                                                      self._mofgbml_args.get("RULE_CHANGE_RT"),
                                                      self._train,
                                                      self._knowledge,
                                                      self._mofgbml_args.get("MAX_NUM_RULES"),
                                                      self._random_gen,
                                                      self._mofgbml_args.get("MICHIGAN_CROSS_RT"),
                                                  ),
                                                  pittsburgh_crossover,
                                                  crossover_probability)
        elif self._mofgbml_args.get("CROSSOVER_TYPE") == "pittsburgh-crossover":
            self._crossover = pittsburgh_crossover
        else:
            raise ValueError("Unknown crossover type")

        num_objectives_michigan = 2
        num_constraints_michigan = 0

        num_vars_pittsburgh = self._mofgbml_args.get("INITIATION_RULE_NUM")
        num_constraints_pittsburgh = 0

        michigan_solution_builder = MichiganSolutionBuilder(self._random_gen,
                                                            num_objectives_michigan,
                                                            num_constraints_michigan,
                                                            self._rule_builder)

        # classification = SingleWinnerRuleSelection(self._mofgbml_args.get("CACHE_SIZE"))
        classification = SingleWinnerRuleSelection()

        self._problem = PittsburghProblem(num_vars_pittsburgh,
                                          self._objectives,
                                          num_constraints_pittsburgh,
                                          self._train,
                                          michigan_solution_builder,
                                          classification)

    @staticmethod
    def plot_line_interpretability_error_rate_tradeoff(solutions, file_path=None, title=None, xlim=None, grid=True,
                                                       x_key="total_rule_length"):
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
            err_train.append((solution.get_attribute(x_key),
                              solution.get_attribute("training_error_rate")))
            err_test.append((solution.get_attribute(x_key),
                             solution.get_attribute("test_error_rate")))

        PittsburghMain.plot_line_interpretability_error_rate_tradeoff_from_coords(err_train, err_test,
                                                                                       x_label=x_label,
                                                                                       y_label="Error rate",
                                                                                       file_path=file_path, title=title,
                                                                                       xlim=xlim, grid=grid)

    @staticmethod
    def plot_line_interpretability_error_rate_tradeoff_from_coords(err_train, err_test, x_label='Total rule length',
                                                                   y_label='Error rate', file_path=None, title=None,
                                                                   xlim=None, grid=True):
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
            plt.plot(err_train[:, 0], err_train[:, 1], c='darkorange', marker='o', label="Train")
        if len(err_test) != 0:
            plt.plot(err_test[:, 0], err_test[:, 1], c='blue', marker='o', label="Test")
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
