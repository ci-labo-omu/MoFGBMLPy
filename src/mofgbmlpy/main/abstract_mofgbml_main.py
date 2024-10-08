import xml.etree.cElementTree as xml_tree
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from pymoo.core.population import Population
from pymoo.termination import get_termination

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mofgbmlpy.data.input import Input
from mofgbmlpy.data.output import Output

import os
from pymoo.visualization.scatter import Scatter
from importlib import import_module

from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_multi import LearningMulti
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.fuzzy.rule.rule_builder_multi import RuleBuilderMulti
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.utility.util import dash_case_to_class_name, dash_case_to_snake_case


class AbstractMoFGBMLMain(ABC):
    """Abstract MoFGBML Runner

    Attributes:
        _mofgbml_args (Arguments): Arguments loader
        _knowledge_factory_class (AbstractKnowledgeFactory): Class of the knowledge factory
        _knowledge (Knowledge): Knowledge base
        _train (Dataset): Training dataset
        _test (Dataset): test dataset
        _problem (problem): Problem object used by Pymoo
        _objectives (ObjectiveFunction[]): Objective functions
        _termination (Termination): Termination criterion used by Pymoo
        _crossover (Crossover): Crossover operator
        _verbose (bool): If true then display more text (e.g. Pymoo progress)
        _random_gen (numpy.random.Generator): Random generator
        _is_multi_label (bool): If true then the data set is a multi label one
        _learner (AbstractLearning): Learner used to create consequent objects for rules
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

    def load_args(self, args, train=None, test=None):
        """Load the arguments

        Args:
            args (list): List of dash-case arguments
            train (): Training dataset
            test (): Test dataset
        """
        # set command arguments
        self._mofgbml_args.load(args)
        self._random_gen = np.random.Generator(np.random.MT19937(seed=self._mofgbml_args.get("RAND_SEED")))

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

        # Run the algo
        self._objectives = []
        for obj_key in self._mofgbml_args.get("OBJECTIVES"):
            class_name = dash_case_to_class_name(obj_key)
            module_name = "mofgbmlpy.gbml.objectives.pittsburgh." + dash_case_to_snake_case(obj_key)
            imported_module = import_module(module_name)
            objective_class = getattr(imported_module, class_name)

            if obj_key == "error-rate":
                self._objectives.append(objective_class(self._train))
            else:
                self._objectives.append(objective_class())

        antecedent_factory = None
        antecedent_factory_name = self._mofgbml_args.get("ANTECEDENT_FACTORY")
        if antecedent_factory_name == "all-combination-antecedent-factory":
            antecedent_factory = AllCombinationAntecedentFactory(self._random_gen, self._knowledge)
        elif antecedent_factory_name == "heuristic-antecedent-factory":
            antecedent_factory = HeuristicAntecedentFactory(self._train,
                                                            self._knowledge,
                                                            self._mofgbml_args.get("IS_PROBABILITY_DONT_CARE"),
                                                            self._mofgbml_args.get("DONT_CARE_RT"),
                                                            self._mofgbml_args.get(
                                                                "ANTECEDENT_NUMBER_DO_NOT_DONT_CARE"),
                                                            self._random_gen)
        else:
            Exception("Unsupported antecedent factory")

        if self._mofgbml_args.has_key("TERMINATE_EVALUATION") and self._mofgbml_args.get(
                "TERMINATE_EVALUATION") is not None:
            self._termination = get_termination("n_eval", self._mofgbml_args.get("TERMINATE_EVALUATION"))
        elif self._mofgbml_args.has_key("TERMINATE_GENERATION") and self._mofgbml_args.get(
                "TERMINATE_GENERATION") is not None:
            self._termination = get_termination("n_gen", self._mofgbml_args.get("TERMINATE_GENERATION"))
        else:
            raise ValueError("Termination criterion not given or not recognized")

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

        if self._is_multi_label:
            self._learner = LearningMulti(self._train)
            rule_builder = RuleBuilderMulti(antecedent_factory,
                                            self._learner,
                                            self._knowledge)
        else:
            self._learner = LearningBasic(self._train)
            rule_builder = RuleBuilderBasic(antecedent_factory,
                                            self._learner,
                                            self._knowledge)

        michigan_solution_builder = MichiganSolutionBuilder(self._random_gen,
                                                            num_objectives_michigan,
                                                            num_constraints_michigan,
                                                            rule_builder)

        # classification = SingleWinnerRuleSelection(self._mofgbml_args.get("CACHE_SIZE"))
        classification = SingleWinnerRuleSelection()

        self._problem = PittsburghProblem(num_vars_pittsburgh,
                                          self._objectives,
                                          num_constraints_pittsburgh,
                                          self._train,
                                          michigan_solution_builder,
                                          classification)

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

    @abstractmethod
    def run(self):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """
        pass

    def save_results_to_files(self, res):
        """Save the results to CSV and XML files

        Args:
            res (pymoo.core.result.Result):
        """
        non_dominated_solutions = res.opt.get("X")[:, 0]
        archive_solutions = res.non_dominated_archive.get("X")[:, 0]

        results_data = AbstractMoFGBMLMain.solutions_list_to_dict_array(non_dominated_solutions)
        Output.save_data(results_data, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'results.csv')))

        results_data = AbstractMoFGBMLMain.solutions_list_to_dict_array(archive_solutions)
        Output.save_data(results_data,
                         str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'resultsARC.csv')))

        pretty_xml = False
        if self._mofgbml_args is not None and self._mofgbml_args.has_key("PRETTY_XML") and self._mofgbml_args.get(
                "PRETTY_XML"):
            pretty_xml = True

        results_xml = self.get_results_xml(self._knowledge, res.pop)
        Output.save_data(results_xml, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'results.xml')),
                         pretty_xml=pretty_xml)

        Output.writeln(str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'exec_time.txt')), f"{res.exec_time}")

    def main(self, args, train=None, test=None):
        """Main function of the runner

        Args:
            args (list): List of dash-case arguments
            train (Dataset): Training dataset
            test (Dataset): Test dataset

        Returns:
            pymoo.core.result.Result: Results of the run
        """
        # TODO: print information

        self.load_args(args, train, test)

        res = self.run()
        exec_time = res.exec_time

        if self._mofgbml_args.get("VERBOSE"):
            print("Execution time: ", exec_time)

        res.objectives_name = [str(obj) for obj in self._objectives]

        # Keep only non dominated solutions
        non_dominated_mask = NonDominatedSorting().do(res.opt.get("F"), only_non_dominated_front=True)
        res.opt = res.opt[non_dominated_mask]

        self.create_and_add_archives(res)

        # We use archive since it contains all solutions of all populations without filter
        self.update_results_data(res.archive.get("X")[:, 0], self._knowledge, self._train, self._test)
        self.update_results_data(res.pop.get("X")[:, 0], self._knowledge, self._train, self._test,
                                 id_start=len(res.archive))

        if not self._mofgbml_args.get("NO_OUTPUT_FILES"):
            self.save_results_to_files(res)

        # print(res.history[0].pop.get("X"))

        if self._mofgbml_args.get("GEN_PLOT"):
            pareto_front_plot = self.get_pareto_front_plot(res.opt)
            pareto_front_plot.show()
            pareto_front_plot.save(str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'pareto_front.png')))

            # self.save_video(res.history, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'mofgbml.mp4')))
            AbstractMoFGBMLMain.plot_line_interpretability_error_rate_tradeoff(res.opt.get("X")[:, 0],
                                                         str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"),
                                                                          'error_rate_interpretability_tradeoff.png')))
        return res

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
            sol.set_attribute("training_error_rate", sol.get_error_rate(train))
            sol.set_attribute("test_error_rate", sol.get_error_rate(test))
            sol.set_attribute("num_rules", sol.get_num_vars())

            sol_id += 1

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

    @staticmethod
    def plot_line_interpretability_error_rate_tradeoff(solutions, file_path=None, title=None, xlim=None, grid=True, x_key="total_rule_length"):
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

        AbstractMoFGBMLMain.plot_line_interpretability_error_rate_tradeoff_from_coords(err_train, err_test, x_label=x_label, y_label="Error rate", file_path=file_path, title=title, xlim=xlim, grid=grid)

    @staticmethod
    def plot_line_interpretability_error_rate_tradeoff_from_coords(err_train, err_test, x_label='Total rule length', y_label='Error rate', file_path=None, title=None, xlim=None, grid=True):
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

    def plot_fuzzy_variables(self):
        """Plot the fuzzy variables of the knowledge base """
        self._knowledge.plot_fuzzy_variables()

    def show_args(self):
        """Show MoFGBML arguments"""
        print(str(self._mofgbml_args))

    def get_args(self):
        """Get MoFGBML arguments

        Returns:
            Arguments: MoFGBML arguments
        """
        return self._mofgbml_args

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
