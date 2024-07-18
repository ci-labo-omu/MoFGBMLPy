from pymoo.termination import get_termination

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.mo_archive_without_sorting import MoArchiveWithoutSorting
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory

from mofgbmlpy.gbml.operator.repair.pittsburgh_repair import PittsburghRepair
from mofgbmlpy.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.classifier.classifier import Classifier
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain
from mofgbmlpy.main.basic.mofgbml_basic_args import MoFGBMLBasicArgs
import sys

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling


class MoFGBMLBasicMain(AbstractMoFGBMLMain):
    def __init__(self, knowledge_factory_class):
        super().__init__(MoFGBMLBasicArgs(), MoFGBMLBasicMain.hybrid_style_mofgbml, knowledge_factory_class)

    @staticmethod
    def hybrid_style_mofgbml(train, args, knowledge, objectives):
        num_objectives_michigan = 1
        num_constraints_michigan = 0

        num_vars_pittsburgh = args.get("INITIATION_RULE_NUM")
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
                                    objectives,
                                    num_constraints_pittsburgh,
                                    train,
                                    michigan_solution_builder,
                                    classifier)

        crossover_probability = args.get("HYBRID_CROSS_RT")

        algorithm = NSGA2(pop_size=args.get("POPULATION_SIZE"),
                          sampling=HybridGBMLSampling(train),
                          # crossover=PittsburghCrossover(args.get("MIN_NUM_RULES"),
                          #                               args.get("MAX_NUM_RULES")),
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
                          eliminate_duplicates=False,
                          archive=MoArchiveWithoutSorting(duplicate_elimination=False,
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
    runner = MoFGBMLBasicMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(sys.argv[1:])
