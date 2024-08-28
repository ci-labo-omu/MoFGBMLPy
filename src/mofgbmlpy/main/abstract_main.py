import xml.etree.cElementTree as xml_tree
import os
from abc import ABC, abstractmethod

import numpy as np
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
from mofgbmlpy.main.arguments.arguments import Arguments
from mofgbmlpy.main.michigan.michigan_main import MichiganMain
from mofgbmlpy.utility.util import get_algo, dash_case_to_snake_case


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
    def __init__(self, algo_name, problem, pymoo_rand_seed, callback, verbose,
                 is_multi_label, train_file_path, test_file_path, no_output_files, mofgbml_args,
                 experiment_id_dir, root_folder, pop_size, sampling, crossover, repair, mutation,
                 terminate_generation=None, terminate_evaluation=None):
        """Constructor

        Args:
            algo_name (str): name of the algorithm to run (e.g. nsga2)
            problem (Problem): Problem object used by Pymoo
            pymoo_rand_seed (int): Seed for random generation for pymoo
            verbose (bool): If true then display more text (e.g. Pymoo progress)
            callback (Callback): Callback function called after each generation in Pymoo
        """
        self._algo_name = algo_name
        self._problem = problem
        self._pymoo_rand_seed = pymoo_rand_seed
        self._verbose = verbose
        self._callback = callback
        self._experiment_id_dir = experiment_id_dir
        self._root_folder = root_folder

        self._train = Input.input_data_set(train_file_path, is_multi_label)
        self._test = Input.input_data_set(test_file_path, is_multi_label)

        self._terminate_evaluation = terminate_evaluation
        self._terminate_generation = terminate_generation

        algo_kwargs = {
            "pop_size": pop_size,
            "sampling": sampling,
            "crossover": crossover,
            "repair": repair,
            "mutation": mutation
        }

        self._algo = get_algo(algo_name, **algo_kwargs)
        self._mofgbml_args = mofgbml_args

        if self._mofgbml_args is None:
            self._mofgbml_args = self.load_args()

        if self._terminate_evaluation is not None:
            self._termination = get_termination("n_eval", self._terminate_evaluation)
        elif self._terminate_generation is not None:
            self._termination = get_termination("n_gen", self._terminate_generation)
        else:
            raise ValueError("Termination criterion not given or not recognized")

    def run(self):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """

        # Save params
        if not no_output_files:
            Output.mkdirs(self._experiment_id_dir)
            Output.mkdirs(self._root_folder)
            file_name = str(os.path.join(experiment_id_dir, "Consts.txt"))
            Output.writeln(file_name, str(self._mofgbml_args), False)

        res = minimize(self._problem,
                       self._algo,
                       termination=self._termination,
                       seed=self._pymoo_rand_seed,
                       callback=self._callback,
                       verbose=self._verbose)

        return res

    @staticmethod
    def get_antecedent_factory(antecedent_factory_name):
        if antecedent_factory_name == "all-combination-antecedent-factory":
            return AllCombinationAntecedentFactory(self._random_gen, self._knowledge)
        elif antecedent_factory_name == "heuristic-antecedent-factory":
            return HeuristicAntecedentFactory(self._train,
                                              self._knowledge,
                                              self._mofgbml_args.get("IS_PROBABILITY_DONT_CARE"),
                                              self._mofgbml_args.get("DONT_CARE_RT"),
                                              self._mofgbml_args.get(
                                                  "ANTECEDENT_NUMBER_DO_NOT_DONT_CARE"),
                                              self._random_gen)
        else:
            Exception("Unsupported antecedent factory")

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
            solution (PittsburghSolution): Solution evaluated
        """
        solutions = np.array([[solution]], object)
        self._problem.evaluate(solutions)

    def load_args(self):
        args = {}
        for arg in self._mofgbml_args.get_accepted_arguments():
            try:
                args[arg] = getattr(self, "_"+dash_case_to_snake_case(arg))
            except: continue

        self._mofgbml_args.load(args)
