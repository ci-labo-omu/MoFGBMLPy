import xml.etree.cElementTree as xml_tree
import os
from abc import ABC, abstractmethod
from importlib import import_module

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.scatter import Scatter
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video

from mofgbmlpy.data.input import Input
from mofgbmlpy.data.output import Output
from mofgbmlpy.exception.abstract_method_exception import AbstractMethodException
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_multi import LearningMulti
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.fuzzy.rule.rule_builder_multi import RuleBuilderMulti
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.main.arguments.arguments import Arguments
from mofgbmlpy.main.arguments.pittsburgh_style_arguments import PittsburghStyleArguments
from mofgbmlpy.main.michigan.michigan_main import MichiganMain
from mofgbmlpy.utility.util import get_algo, dash_case_to_snake_case, dash_case_to_class_name


class AbstractMain(ABC):
    """Abstract MoFGBML Runner

    Attributes:
        _algo_name (str): name of the algorithm to run (e.g. nsga2)
        _problem (Problem): Problem object used by Pymoo
        _termination (Termination): Termination criterion used by Pymoo
        _pymoo_rand_seed (int): Seed for random generation for pymoo
        _verbose (bool): If true then display more text (e.g. Pymoo progress)
        _callback (Callback): Callback function called after each generation in Pymoo
    """

    def __init__(self, mofgbml_args, knowledge_factory_class):
        """Constructor

        Args:
            mofgbml_args (Arguments): Arguments loader
            knowledge_factory_class (AbstractKnowledgeFactory): Class of the knowledge factory
        """
        self._mofgbml_args = mofgbml_args
        self._knowledge_factory_class = knowledge_factory_class
        self._knowledge = None
        self._train = None
        self._test = None
        self._problem = None
        self._objectives = None
        self._termination = None
        self._crossover = None
        self._verbose = None
        self._random_gen = None
        self._is_multi_label = None
        self._learner = None
        self._callback = None
        self._is_michigan_style = None
        self._repair = None
        self._mutation = None
        self._sampling = None
        self._algo = None
        self._pop_size = None
        self._antecedent_factory = None
        self._rule_builder = None
        self._pymoo_rand_seed = None

    def load_args(self, args, train=None, test=None):
        """Load the arguments

        Args:
            args (list): List of dash-case arguments
            train (Dataset): Training dataset
            test (Dataset): Test dataset
        """
        # Add parameters specific to the algorithm used
        algo_name = args[args.index("--algorithm") + 1]
        self._mofgbml_args.load_config_file(algo_name + "_arguments")

        # load command arguments
        self._mofgbml_args.load(args)

        seed = self._mofgbml_args.get("RAND_SEED")
        self._random_gen = np.random.Generator(np.random.MT19937(seed=seed))
        self._pymoo_rand_seed = seed

        self._verbose = self._mofgbml_args.get("VERBOSE")

        # Save params
        if not self._mofgbml_args.get("NO_OUTPUT_FILES"):
            Output.mkdirs(self._mofgbml_args.get("EXPERIMENT_ID_DIR"))
            Output.mkdirs(self._mofgbml_args.get("ROOT_FOLDER"))
            file_name = str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), "Consts.txt"))
            Output.writeln(file_name, str(self._mofgbml_args), False)

        # Load dataset
        if train is not None and test is not None:
            self._train, self._test = train, test
        else:
            self._train, self._test = Input.get_train_test_files(self._mofgbml_args)

        self._is_multi_label = self._mofgbml_args.get("IS_MULTI_LABEL")

        # Create knowledge object
        self._knowledge = self._knowledge_factory_class(self._train.get_num_dim()).create()
        self._pop_size = self._mofgbml_args.get("POPULATION_SIZE")

        # Load objectives
        self._objectives = self._get_objectives(isinstance(self._mofgbml_args, PittsburghStyleArguments))

        self._rule_builder = self._get_rule_builder()
        self._termination = self._get_termination()

        self._load_additional_args()

        self._algo = self.get_pymoo_algo()

    def run(self):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """

        # Save params
        if not self._mofgbml_args.get("NO_OUTPUT_FILES"):
            Output.mkdirs(self._mofgbml_args.get("EXPERIMENT_ID_DIR"))
            Output.mkdirs(self._mofgbml_args.get("ROOT_FOLDER"))
            file_name = str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), "Consts.txt"))
            Output.writeln(file_name, str(self._mofgbml_args), False)

        res = minimize(self._problem,
                       self._algo,
                       termination=self._termination,
                       seed=self._pymoo_rand_seed,
                       callback=self._callback,
                       verbose=self._verbose)

        return res

    def _get_antecedent_factory(self):
        antecedent_factory_name = self._mofgbml_args.get("ANTECEDENT_FACTORY")
        if antecedent_factory_name == "all-combination-antecedent-factory":
            return AllCombinationAntecedentFactory(knowledge=self._knowledge, random_gen=self._random_gen)
        elif antecedent_factory_name == "heuristic-antecedent-factory":
            return HeuristicAntecedentFactory(training_set=self._train,
                                              knowledge=self._knowledge,
                                              is_dc_probability=self._mofgbml_args.get("IS_PROBABILITY_DONT_CARE"),
                                              dc_rate=self._mofgbml_args.get("DONT_CARE_RT"),
                                              antecedent_number_do_not_dont_care=self._mofgbml_args.get(
                                                  "ANTECEDENT_NUMBER_DO_NOT_DONT_CARE"),
                                              random_gen=self._random_gen)
        else:
            Exception("Unsupported antecedent factory")

    def _get_termination(self):
        if self._mofgbml_args.has_key("TERMINATE_EVALUATION") and self._mofgbml_args.get(
                "TERMINATE_EVALUATION") is not None:
            return get_termination("n_eval", self._mofgbml_args.get("TERMINATE_EVALUATION"))
        elif self._mofgbml_args.has_key("TERMINATE_GENERATION") and self._mofgbml_args.get(
                "TERMINATE_GENERATION") is not None:
            return get_termination("n_gen", self._mofgbml_args.get("TERMINATE_GENERATION"))
        else:
            raise ValueError("Termination criterion not given or not recognized")

    def _get_objectives(self, is_pittsburgh_style):
        objectives = []
        module_base = f"mofgbmlpy.gbml.objectives.{'pittsburgh' if is_pittsburgh_style else 'michigan'}."

        for obj_key in self._mofgbml_args.get("OBJECTIVES"):
            class_name = dash_case_to_class_name(obj_key)
            module_name = module_base + dash_case_to_snake_case(obj_key)
            imported_module = import_module(module_name)
            objective_class = getattr(imported_module, class_name)

            if obj_key == "error-rate":
                objectives.append(objective_class(self._train))
            else:
                objectives.append(objective_class())
        return objectives

    def _get_rule_builder(self):
        antecedent_factory = self._get_antecedent_factory()
        if self._is_multi_label:
            self._learner = LearningMulti(self._train)
            return RuleBuilderMulti(antecedent_factory,
                                    self._learner,
                                    self._knowledge)
        else:
            self._learner = LearningBasic(self._train)
            return RuleBuilderBasic(antecedent_factory,
                                    self._learner,
                                    self._knowledge)

    def get_pymoo_algo(self):
        algo_name = self._mofgbml_args.get("ALGORITHM")

        algo_args = {
            "eliminate_duplicates": False,
            "save_history": True,
            "pop_size": self._pop_size,
            "sampling": self._sampling,
            "crossover": self._crossover,
            "repair": self._repair,
            "mutation": self._mutation
        }

        algos = {
            "nsga2": {"class": NSGA2, "additional_args": ["n_offsprings"]},
            "nsga3": {"class": NSGA3, "additional_args": ["n_offsprings"]},
            "moead": {"class": MOEAD, "additional_args": [
                "neighborhood_selection_probability",
                "neighborhood_size",
                "offspring_population_size"
            ]},
        }

        if algo_name not in algos:
            raise ValueError("Unknown algo name")

        for arg in algos[algo_name]["additional_args"]:
            algo_args[arg] = self._mofgbml_args.get(arg.upper())

        return algos[algo_name]["class"](algo_args)

    def save_results_to_files(self, res):
        """Save the results to CSV and XML files

        Args:
            res (pymoo.core.result.Result):
        """
        non_dominated_solutions = res.opt.get("X")[:, 0]
        archive_solutions = res.non_dominated_archive.get("X")[:, 0]

        results_data = AbstractMain.solutions_list_to_dict_array(non_dominated_solutions)
        Output.save_data(results_data, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'results.csv')))

        results_data = AbstractMain.solutions_list_to_dict_array(archive_solutions)
        Output.save_data(results_data,
                         str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'resultsARC.csv')))

        pretty_xml = False
        if self._mofgbml_args is not None and self._mofgbml_args.has_key("PRETTY_XML") and self._mofgbml_args.get(
                "PRETTY_XML"):
            pretty_xml = True

        results_xml = self.get_results_xml(self._knowledge, res.pop)
        Output.save_data(results_xml, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'results.xml')),
                         pretty_xml=pretty_xml)

        Output.writeln(str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'exec_time.txt')),
                       f"{res.exec_time}")

    @staticmethod
    def solutions_list_to_dict_array(solutions):
        """Convert a list of solutions to an array of dictionaries of their attributes (error rate, ...)

        Args:
            solutions (AbstractSolution[]): List of solutions

        Returns:
            dict[]: Array of dictionaries of attributes
        """
        results_data = np.zeros(len(solutions), dtype=object)
        for i in range(len(solutions)):
            results_data[i] = solutions[i].get_attributes()
        return results_data

    @staticmethod
    def create_and_add_archives(res):
        """Create and add an archive and non dominated archive to the result object

        Args:
            res (pymoo.core.result.Result): Result object
        """
        # Create archive population from history and apply non dominated sorting
        res.archive = Population.empty()
        for i in range(len(res.history)):
            res.archive = Population.merge(res.archive, res.history[i].pop)

        archive_objectives = res.archive.get("F")
        non_dominated_mask = NonDominatedSorting().do(archive_objectives, only_non_dominated_front=True)
        res.non_dominated_archive = res.archive[non_dominated_mask]

    @staticmethod
    def update_results_data(solutions, knowledge, train, test, id_start=0):
        """Update the solutions data (attributes)

        Args:
            solutions (PittsburghSolution[]): solutions
            knowledge (Knowledge): Knowledge base
            train (Dataset): Training dataset
            test (Dataset):  Test dataset
            id_start (int): The ID is determined by the order of the solutions in loop. This parameter determines the starting value for the ID
        """
        raise AbstractMethodException()

    def get_results_xml(self, knowledge, pop):
        """Get the results as an XML object

        Args:
            knowledge (Knowledge): Knowledge base
            pop (Population): Population of solutions

        Returns:
            xml.etree.cElementTree.ElementTree: XML element
        """
        root = xml_tree.Element("results")
        root.append(self._mofgbml_args.to_xml())
        root.append(knowledge.to_xml())
        population = xml_tree.SubElement(root, "population")
        for ind in pop:
            population.append(ind.X[0].to_xml())

        return xml_tree.ElementTree(root)

    @staticmethod
    def save_video(history, filename):
        """Save a video of the history, showing the progression for each generation

        Args:
            history (Population[]): List of Pymoo populations
            filename (str): Name of the file where the video will be saved
        """
        with Recorder(Video(filename)) as rec:
            for i in range(len(history)):
                sc = Scatter(title=(f"Gen {i + 1}"))
                sc.add(history[i].pop.get("F"))
                sc.do()

                rec.record()

    def get_pareto_front_plot(self, population):
        """Get the Pareto front plot

        Args:
            population (Population): Population of solutions

        Returns:
             pymoo.visualization.scatter.Scatter: Pareto front plot
        """
        objectives = population.get("F")

        if objectives.shape[1] <= 1:
            raise ValueError("At least 2 objectives are required to plot")

        plot = Scatter(labels=self._mofgbml_args.get("OBJECTIVES"))
        plot.add(objectives, color="red")
        # In a Notebook the plot doesn't show if we don't use show directly inside,
        # hence we have to return the plot instead of showing it
        return plot

    def plot_fuzzy_variables(self):
        """Plot the fuzzy variables of the knowledge base """
        self._knowledge.plot_fuzzy_variables()

    def get_train_set(self):
        """Get the training set

        Returns:
            Dataset: Training set

        """
        return self._train

    def get_test_set(self):
        """Get the test set

        Returns:
            Dataset: Test set

        """
        return self._test

    def evaluate(self, solution):
        """Evaluate the solution. The results will be in the solution objectives attribute

        Args:
            solution (AbstractSolution): Solution evaluated
        """
        solutions = np.array([[solution]], object)
        self._problem.evaluate(solutions)

    def _load_additional_args(self):
        """Load either Michigan or Pittsburgh approach arguments (problem, ...) """
        raise AbstractMethodException()
