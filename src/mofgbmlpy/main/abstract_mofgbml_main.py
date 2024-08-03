import xml.etree.cElementTree as xml_tree
from abc import ABC

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

import random


from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video

from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.utility.util import dash_case_to_class_name, dash_case_to_snake_case


class AbstractMoFGBMLMain(ABC):
    _mofgbml_args = None
    _algo = None
    _knowledge_factory_class = None
    __knowledge = None

    def __init__(self, mofgbml_args, algo, knowledge_factory_class):
        self._mofgbml_args = mofgbml_args
        self._algo = algo
        self._knowledge_factory_class = knowledge_factory_class

    def main(self, args):
        # TODO: print information

        # set command arguments
        self._mofgbml_args.load(args)
        random.seed(self._mofgbml_args.get("RAND_SEED"))
        np.random.seed(self._mofgbml_args.get("RAND_SEED"))

        Output.mkdirs(self._mofgbml_args.get("ROOT_FOLDER"))

        # Save params
        file_name = str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), "Consts.txt"))
        Output.writeln(file_name, str(self._mofgbml_args), False)

        # Load dataset
        train, test = Input.get_train_test_files(self._mofgbml_args)

        # Create knowledge object
        self.__knowledge = self._knowledge_factory_class(train.get_num_dim()).create()

        # Run the algo
        objectives = []
        for obj_key in self._mofgbml_args.get("OBJECTIVES"):
            class_name = dash_case_to_class_name(obj_key)
            module_name = "mofgbmlpy.gbml.objectives.pittsburgh." + dash_case_to_snake_case(obj_key)
            imported_module = import_module(module_name)
            objective_class = getattr(imported_module, class_name)

            if obj_key == "error-rate":
                objectives.append(objective_class(train))
            else:
                objectives.append(objective_class())

        antecedent_factory_name = self._mofgbml_args.get("ANTECEDENT_FACTORY")
        if antecedent_factory_name == "all-combination-antecedent-factory":
            antecedent_factory = AllCombinationAntecedentFactory(self.__knowledge)
        elif antecedent_factory_name == "heuristic-antecedent-factory":
            antecedent_factory = HeuristicAntecedentFactory(train,
                                                            self.__knowledge,
                                                            self._mofgbml_args.get("IS_PROBABILITY_DONT_CARE"),
                                                            self._mofgbml_args.get("DONT_CARE_RT"),
                                                            self._mofgbml_args.get("ANTECEDENT_NUMBER_DO_NOT_DONT_CARE"))
        else:
            Exception("Unsupported antecedent factory")

        if self._mofgbml_args.has_key("TERMINATE_EVALUATION") and self._mofgbml_args.get(
                "TERMINATE_EVALUATION") is not None:
            termination = get_termination("n_eval", self._mofgbml_args.get("TERMINATE_EVALUATION"))
        elif self._mofgbml_args.has_key("TERMINATE_GENERATION") and self._mofgbml_args.get(
                "TERMINATE_GENERATION") is not None:
            termination = get_termination("n_gen", self._mofgbml_args.get("TERMINATE_GENERATION"))
        else:
            raise Exception("Termination criterion not given or not recognized")

        pittsburgh_crossover = PittsburghCrossover(self._mofgbml_args.get("MIN_NUM_RULES"),
                                        self._mofgbml_args.get("MAX_NUM_RULES"),
                                        self._mofgbml_args.get("PITTSBURGH_CROSS_RT"))

        if self._mofgbml_args.get("CROSSOVER_TYPE") == "hybrid-gbml-crossover":
            crossover_probability = self._mofgbml_args.get("HYBRID_CROSS_RT")
            crossover = HybridGBMLCrossover(self._mofgbml_args.get("MICHIGAN_OPE_RT"),
                                            MichiganCrossover(
                                                self._mofgbml_args.get("RULE_CHANGE_RT"),
                                                train,
                                                self.__knowledge,
                                                self._mofgbml_args.get("MAX_NUM_RULES"),
                                                self._mofgbml_args.get("MICHIGAN_CROSS_RT")
                                            ),
                                            pittsburgh_crossover,
                                            crossover_probability)
        elif self._mofgbml_args.get("CROSSOVER_TYPE") == "pittsburgh-crossover":
            crossover = pittsburgh_crossover

        res = self._algo(train, self._mofgbml_args, self.__knowledge, objectives, termination, antecedent_factory, crossover)
        exec_time = res.exec_time

        print("Execution time: ", exec_time)

        non_dominated_solutions = res.X

        res.archive = Population.empty()
        for i in range(len(res.history)):
            res.archive = Population.merge(res.archive, res.history[i].pop)

        archive_objectives = res.archive.get("F")
        non_dominated_mask = NonDominatedSorting().do(archive_objectives, only_non_dominated_front=True)
        non_dominated_archive_pop = res.archive[non_dominated_mask]

        archive_solutions = np.empty((len(non_dominated_archive_pop), res.X.shape[1]), dtype=object)
        for i in range(len(non_dominated_archive_pop)):
            archive_solutions[i] = non_dominated_archive_pop[i].X

        results_data = AbstractMoFGBMLMain.get_results_data(non_dominated_solutions, self.__knowledge, train, test)
        Output.save_data(results_data, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'results.csv')))

        results_data = AbstractMoFGBMLMain.get_results_data(archive_solutions, self.__knowledge, train, test)
        Output.save_data(results_data,
                            str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'resultsARC.csv')))

        results_xml = self.get_results_xml(self.__knowledge, res.pop)
        Output.save_data(results_xml, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'results.xml')),
                            args=self._mofgbml_args)
        res.objectives_name = [str(obj) for obj in objectives]

        if self._mofgbml_args.get("GEN_PLOT"):
            pareto_front_plot = self.get_pareto_front_plot(res.opt)
            pareto_front_plot.show()
            pareto_front_plot.save(str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'pareto_front.png')))

            # self.save_video(res.history, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'mofgbml.mp4')))
            self.plot_line_interpretability_acc_tradeoff(res.opt, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'accuracy_interpretability_tradeoff.png')))

        return res

    @staticmethod
    def get_results_data(solutions, knowledge, train, test):
        results_data = np.zeros(len(solutions), dtype=object)
        for i in range(len(solutions)):
            sol = solutions[i][0]
            if sol.get_num_vars() != 0 and sol.get_var(0).get_num_vars() != 0:
                total_coverage = 1
            else:
                total_coverage = 0
            for rule_i in range(sol.get_num_vars()):
                michigan_solution = sol.get_var(rule_i)
                fuzzy_set_indices = michigan_solution.get_vars()
                for dim_i in range(len(fuzzy_set_indices)):
                    total_coverage *= knowledge.get_support(dim_i, fuzzy_set_indices[dim_i])

            results_data[i] = {}
            results_data[i]["id"] = i
            results_data[i]["total_coverage"] = total_coverage
            results_data[i]["total_rule_length"] = sol.get_total_rule_length()
            results_data[i]["average_rule_weight"] = sol.get_average_rule_weight()
            results_data[i]["training_error_rate"] = sol.get_error_rate(train)
            results_data[i]["test_error_rate"] = sol.get_error_rate(test)
            results_data[i]["num_rules"] = sol.get_num_vars()

            sol.set_attribute("id", results_data[i]["id"])
            sol.set_attribute("total_coverage", results_data[i]["total_coverage"])
            sol.set_attribute("total_rule_length", results_data[i]["total_rule_length"])
            sol.set_attribute("average_rule_weight", results_data[i]["average_rule_weight"])
            sol.set_attribute("training_error_rate", results_data[i]["training_error_rate"])
            sol.set_attribute("test_error_rate", results_data[i]["test_error_rate"])
            sol.set_attribute("num_rules", results_data[i]["num_rules"])
        return results_data

    def get_results_xml(self, knowledge, pop):
        root = xml_tree.Element("results")
        root.append(self._mofgbml_args.to_xml())
        root.append(knowledge.to_xml())
        population = xml_tree.SubElement(root, "population")
        for ind in pop:
            population.append(ind.X[0].to_xml())

        return xml_tree.ElementTree(root)

    def save_video(self, history, filename):
        with Recorder(Video(filename)) as rec:
            for i in range(len(history)):
                sc = Scatter(title=(f"Gen {i+1}"))
                sc.add(history[i].pop.get("F"))
                sc.do()

                rec.record()

    def get_pareto_front_plot(self, population):
        objectives = population.get("F")

        if objectives.shape[1] <= 1:
            raise Exception("At least 2 objectives are required to plot")

        plot = Scatter(labels=self._mofgbml_args.get("OBJECTIVES"))
        plot.add(objectives, color="red")
        # In a Notebook the plot doesn't show if we don't use show directly inside,
        # hence we have to return the plot instead of showing it
        return plot


    @staticmethod
    def plot_line_interpretability_acc_tradeoff(population, file_path=None):
        acc_train = []
        acc_test = []

        for solution in population.get("X")[:, 0]:
            acc_train.append(
                (solution.get_attribute("total_rule_length"), 1 - solution.get_attribute("training_error_rate")))
            acc_test.append(
                (solution.get_attribute("total_rule_length"), 1 - solution.get_attribute("test_error_rate")))

        acc_train = list(set(acc_train))
        acc_train.sort()
        for i in range(len(acc_train)):
            acc_train[i] = list(acc_train[i])  # tuple to list
        acc_train = np.array(acc_train)

        acc_test = list(set(acc_test))
        acc_test.sort()
        for i in range(len(acc_test)):
            acc_test[i] = list(acc_test[i])  # tuple to list
        acc_test = np.array(acc_test)

        plt.figure(figsize=(10, 6))
        plt.plot(acc_train[:, 0], acc_train[:, 1], c='darkorange', marker='o', label="Train")
        plt.plot(acc_test[:, 0], acc_test[:, 1], c='blue', marker='o', label="Test")
        plt.xlabel('total rule length')

        plt.ylabel('accuracy')
        plt.ylim(0, 1)
        plt.legend(loc="upper left")
        plt.show()

        if file_path is not None:
            plt.savefig(file_path)

    def plot_fuzzy_variables(self):
        self.__knowledge.plot_fuzzy_variables()

    def show_args(self):
        print(str(self._mofgbml_args))
