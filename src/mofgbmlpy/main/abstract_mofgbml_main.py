import xml.etree.cElementTree as xml_tree
from abc import ABC, abstractmethod

import numpy as np
from pymoo.termination import get_termination
from pymoo.util.archive import MultiObjectiveArchive

from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory

from mofgbmlpy.gbml.operator.repair.pittsburgh_repair import PittsburghRepair
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.classifier.classifier import Classifier
from mofgbmlpy.data.input import Input
from mofgbmlpy.data.output import Output
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.main.basic.mofgbml_basic_args import MoFGBMLBasicArgs
import sys
import os
from pymoo.visualization.scatter import Scatter

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import random

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic
from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory import HomoTriangleKnowledgeFactory

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling
from mofgbmlpy.gbml.BasicDuplicateElimination import BasicDuplicateElimination
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video


class AbstractMoFGBMLMain(ABC):
    _mofgbml_args = None
    _algo = None
    _knowledge_factory_class = None

    def __init__(self, mofgbml_args, algo, knowledge_factory_class):
        random.seed(mofgbml_args.get("RAND_SEED"))
        np.random.seed(mofgbml_args.get("RAND_SEED"))

        self._mofgbml_args = mofgbml_args
        self._algo = algo
        self._knowledge_factory_class = knowledge_factory_class

    def main(self, args):
        # TODO: print information

        # set command arguments
        self._mofgbml_args.load(args)

        Output.mkdirs(self._mofgbml_args.get("ROOT_FOLDER"))

        # Save params
        file_name = str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), "Consts.txt"))
        Output.writeln(file_name, str(self._mofgbml_args), False)

        # Load dataset
        train, test = Input.get_train_test_files(self._mofgbml_args)

        # Create knowledge object
        knowledge = self._knowledge_factory_class(train.get_num_dim()).create()

        # Run the algo
        res = self._algo(train, self._mofgbml_args, knowledge)
        exec_time = res.exec_time

        print("Execution time: ", exec_time)

        non_dominated_solutions = res.X
        archive_population = np.empty((len(res.archive), res.X.shape[1]), dtype=object)
        for i in range(len(res.archive)):
            archive_population[i] = res.archive[i].X

        #
        # with Recorder(Video("ga.mp4")) as rec:
        #     # for each algorithm object in the history
        #     for entry in res.history:
        #         sc = Scatter(title=("Gen %s" % entry.n_gen))
        #         sc.add(entry.pop.get("F"))
        #         sc.do()
        #
        #         # finally record the current visualization to the video
        #         rec.record()

        plot_data = np.empty(res.F.shape, dtype=object)
        for i in range(len(res.F)):
            plot_data[i] = [int(res.F[i][1]), res.F[i][0]]

        if not self._mofgbml_args.get("NO_PLOT"):
            plot = Scatter(labels=["Number of rules", "Error rate"])
            plot.add(plot_data, color="red")
            plot.show()

        # f_archive = np.empty((len(res.archive), res.F.shape[1]), dtype=object)
        # for i in range(len(res.archive)):
        #     f_archive[i] = res.archive[i].F[[1, 0]]
        #
        # plot = Scatter()
        # plot.add(f_archive, color="red")
        # plot.show()

        results_data = AbstractMoFGBMLMain.get_results_data(non_dominated_solutions, knowledge, train, test)
        Output.save_results(results_data, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'results.csv')))

        results_data = AbstractMoFGBMLMain.get_results_data(archive_population, knowledge, train, test)
        Output.save_results(results_data, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'resultsARC.csv')))
        
        results_xml = self.get_results_xml(knowledge, res.pop)
        Output.save_results(results_xml, str(os.path.join(self._mofgbml_args.get("EXPERIMENT_ID_DIR"), 'results.xml')), args=self._mofgbml_args)

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
        return results_data

    def get_results_xml(self, knowledge, pop):
        root = xml_tree.Element("results")
        root.append(self._mofgbml_args.to_xml())
        root.append(knowledge.to_xml())
        population = xml_tree.SubElement(root, "population")
        for ind in pop:
            population.append(ind.X[0].to_xml())

        return xml_tree.ElementTree(root)
