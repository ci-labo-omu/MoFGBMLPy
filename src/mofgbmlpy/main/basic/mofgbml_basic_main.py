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


class MoFGBMLBasicMain:
    @staticmethod
    def main(args):
        # TODO: print information

        # Consts.set()
        #...

        mofgbml_basic_args = MoFGBMLBasicArgs()
        Output.mkdirs(mofgbml_basic_args.get("ROOT_FOLDER"))

        # set command arguments to static variables
        mofgbml_basic_args.load(args)

        # Save params
        file_name = str(os.path.join(mofgbml_basic_args.get("EXPERIMENT_ID_DIR"), "Consts.txt"))
        Output.writeln(file_name, str(mofgbml_basic_args), False)

        # Load dataset
        train, test = Input.get_train_test_files(mofgbml_basic_args)

        # Run the algo
        exec_time = MoFGBMLBasicMain.hybrid_style_mofgbml(train, test, mofgbml_basic_args)
        print("Execution time: ", exec_time)

    @staticmethod
    def hybrid_style_mofgbml(train, test, args):
        random.seed(args.get("RAND_SEED"))
        np.random.seed(args.get("RAND_SEED"))
        knowledge = HomoTriangleKnowledgeFactory.create2_3_4_5(train.get_num_dim())
        bounds_michigan = MichiganSolution.make_bounds(knowledge)

        num_objectives_michigan = 1
        num_constraints_michigan = 0

        num_vars_pittsburgh = args.get("INITIATION_RULE_NUM")
        num_objectives_pittsburgh = 2
        num_constraints_pittsburgh = 0

        # rule_builder = RuleBuilderBasic(AllCombinationAntecedentFactory(knowledge),
        #                                 LearningBasic(train),
        #                                 knowledge)
        #
        rule_builder = RuleBuilderBasic(HeuristicAntecedentFactory(train,
                                                                   knowledge,
                                                                   args.get("IS_DONT_CARE_PROBABILITY"),
                                                                   args.get("DONT_CARE_RT"),
                                                                   args.get("ANTECEDENT_NUM_NOT_DONT_CARE")),
                                        LearningBasic(train),
                                        knowledge)

        michigan_solution_builder = MichiganSolutionBuilder(bounds_michigan,
                                                            num_objectives_michigan,
                                                            num_constraints_michigan,
                                                            rule_builder)

        classification = SingleWinnerRuleSelection()
        classifier = Classifier(classification)

        problem = PittsburghProblem(num_vars_pittsburgh,
                                    num_objectives_pittsburgh,
                                    num_constraints_pittsburgh,
                                    train,
                                    michigan_solution_builder,
                                    classifier)

        crossover_probability = args.get("")

        algorithm = NSGA2(pop_size=args.get("POPULATION_SIZE"),
                          sampling=HybridGBMLSampling(train),
                          # crossover=PittsburghCrossover(args.get("MIN_NUM_RULES"),
                          #                               args.get("MAX_NUM_RULES")),
                          crossover=HybridGBMLCrossover(args.get("MICHIGAN_OPE_RT"),
                                                        MichiganCrossover(
                                                            args.get("RULE_CHANGE_RT"),
                                                            train,
                                                            args.get("MICHIGAN_CROSS_RT"),
                                                            knowledge
                                                        ),
                                                        PittsburghCrossover(
                                                            args.get("MIN_NUM_RULES"),
                                                            args.get("MAX_NUM_RULES"),
                                                            args.get("PITTSBURGH_CROSS_RT")),
                                                        crossover_probability),
                          repair=PittsburghRepair(),
                          mutation=PittsburghMutation(train, knowledge),
                          eliminate_duplicates=BasicDuplicateElimination(),
                          archive=MultiObjectiveArchive(duplicate_elimination=BasicDuplicateElimination(),
                                                        max_size=None, truncate_size=None))

        res = minimize(problem,
                       algorithm,
                       # termination=get_termination("n_eval", args.get("TERMINATE_GENERATION")),
                       termination=get_termination("n_eval", args.get("TERMINATE_EVALUATION")),
                       seed=1,
                       # save_history=True,
                       verbose=True)

        # print("\n\nEND\n\n")
        # for i in range(len(res.X)):
        #     print(res.X[i][0])
        #     print(f"({res.F[i][0]}, {res.F[i][0]}) - ({res.X[i][0].get_num_vars()}, {res.F[i][1]})")


        non_dominated_solutions = res.X
        archive_population = np.empty((len(res.archive), res.X.shape[1]), dtype=object)
        for i in range(len(res.archive)):
            archive_population[i] = res.archive[i].X

        exec_time = res.exec_time
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

        results_data = MoFGBMLBasicMain.get_results_data(non_dominated_solutions, knowledge, train, test)
        Output.save_results(results_data, str(os.path.join(args.get("EXPERIMENT_ID_DIR"), 'results.csv')))

        results_data = MoFGBMLBasicMain.get_results_data(archive_population, knowledge, train, test)
        Output.save_results(results_data, str(os.path.join(args.get("EXPERIMENT_ID_DIR"), 'resultsARC.csv')))

        return exec_time

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


if __name__ == '__main__':
    MoFGBMLBasicMain.main(sys.argv[1:])
