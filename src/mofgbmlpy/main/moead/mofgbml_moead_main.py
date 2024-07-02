import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.termination import get_termination
from pymoo.util.archive import MultiObjectiveArchive
from pymoo.util.ref_dirs import get_reference_directions

from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
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
from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain
from mofgbmlpy.main.moead.mofgbml_moead_args import MoFGBMLMOEADArgs
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


class MoFGBMLMOEADMain(AbstractMoFGBMLMain):
    def __init__(self, knowledge_factory_class):
        super().__init__(MoFGBMLMOEADArgs(), MoFGBMLMOEADMain.hybrid_style_mofgbml, knowledge_factory_class)

    @staticmethod
    def hybrid_style_mofgbml(train, args, knowledge):
        num_objectives_michigan = 2
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

        michigan_solution_builder = MichiganSolutionBuilder(num_objectives_michigan,
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

        crossover_probability = args.get("HYBRID_CROSS_RT")

        ref_dirs = get_reference_directions("uniform",
                                            problem.get_num_objectives(),
                                            n_partitions=args.get("POPULATION_SIZE")-1)

        #TODO: check "variation" in the java code

        # Note: if num_obj <=2, pymoo uses Tschebyscheff
        algorithm = MOEAD(
            ref_dirs,
            n_neighbors=args.get("NEIGHBORHOOD_SIZE"),
            prob_neighbor_mating=args.get("NEIGHBORHOOD_SELECTION_PROBABILITY"),
            sampling=HybridGBMLSampling(train),
            crossover=HybridGBMLCrossover(args.get("MICHIGAN_OPE_RT"),
                                          MichiganCrossover(
                                              args.get("RULE_CHANGE_RT"),
                                              train,
                                              knowledge,
                                              args.get("MAX_NUM_RULES"),
                                              args.get("MICHIGAN_CROSS_RT")
                                          ),
                                          PittsburghCrossover(
                                              args.get("MIN_NUM_RULES"),
                                              args.get("MAX_NUM_RULES"),
                                              args.get("PITTSBURGH_CROSS_RT")),
                                          crossover_probability),
            repair=PittsburghRepair(),
            mutation=PittsburghMutation(train, knowledge),
            archive=MultiObjectiveArchive(duplicate_elimination=False,
                                          max_size=None, truncate_size=None))

        res = minimize(problem,
                       algorithm,
                       # termination=get_termination("n_eval", args.get("TERMINATE_GENERATION")),
                       termination=get_termination("n_eval", args.get("TERMINATE_EVALUATION")),
                       seed=1,
                       # save_history=True,
                       verbose=True)
        return res


if __name__ == '__main__':
    runner = MoFGBMLMOEADMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(sys.argv[1:])
