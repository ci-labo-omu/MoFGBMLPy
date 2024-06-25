import numpy as np
from pymoo.termination import get_termination
from pymoo.util.archive import MultiObjectiveArchive

from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
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
        knowledge = HomoTriangleKnowledgeFactory.create2_3_4_5(train.get_num_dim())
        bounds_michigan = MichiganSolution.make_bounds(knowledge)

        num_objectives_michigan = 1
        num_constraints_michigan = 0

        num_vars_pittsburgh = args.get("INITIATION_RULE_NUM")
        num_objectives_pittsburgh = 2
        num_constraints_pittsburgh = 0

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

        crossover_probability = 1

        algorithm = NSGA2(pop_size=args.get("POPULATION_SIZE"),
                          sampling=HybridGBMLSampling(train),
                          crossover=PittsburghCrossover(args.get("MIN_NUM_RULES"),
                                                        args.get("MAX_NUM_RULES")),
                          # crossover=HybridGBMLCrossover(crossover_probability,
                          #                               args.get("MIN_NUM_RULES"),
                          #                               args.get("MAX_NUM_RULES")),
                          mutation=PittsburghMutation(train, knowledge),
                          eliminate_duplicates=BasicDuplicateElimination(),
                          archive=MultiObjectiveArchive(duplicate_elimination=BasicDuplicateElimination(), max_size=None, truncate_size=None))


        res = minimize(problem,
                       algorithm,
                       termination=get_termination("n_eval", args.get("TERMINATE_GENERATION")),
                       # get_termination("n_eval", args.get("TERMINATE_EVALUATION"),
                       seed=1,
                       verbose=True)

        plot = Scatter()
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plot.add(res.F, color="red")
        plot.show()

        non_dominated_solutions = res.pop
        archive_population = res.archive
        exec_time = res.exec_time

        results_data = MoFGBMLBasicMain.get_results_data(non_dominated_solutions, knowledge, train, test)
        Output.save_results(results_data, str(os.path.join(args.get("EXPERIMENT_ID_DIR"), 'results.csv')))

        results_data = MoFGBMLBasicMain.get_results_data(archive_population, knowledge, train, test)
        Output.save_results(results_data, str(os.path.join(args.get("EXPERIMENT_ID_DIR"), 'resultsARC.csv')))

        return exec_time

    @staticmethod
    def get_results_data(solutions, knowledge, train, test):
        results_data = np.zeros(len(solutions), dtype=object)
        for i in range(len(solutions)):
            total_rule_weight = 0
            sol = solutions[i].X[0]
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
            results_data[i]["total_coverage"] = total_coverage
            results_data[i]["total_rule_weight"] = sol.get_total_rule_weight()
            results_data[i]["average_rule_weight"] = total_rule_weight / sol.get_num_vars()
            results_data[i]["training_error_rate"] = sol.get_error_rate(train)
            results_data[i]["test_error_rate"] = sol.get_error_rate(test)
            results_data[i]["num_rules"] = sol.get_num_vars()
        return results_data


if __name__ == '__main__':
    MoFGBMLBasicMain.main(sys.argv[1:])
