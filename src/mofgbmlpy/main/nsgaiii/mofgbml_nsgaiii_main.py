from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
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
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_args import MoFGBMLNSGAIIArgs
import sys

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling
from mofgbmlpy.main.nsgaiii.mofgbml_nsgaiii_args import MoFGBMLNSGAIIIArgs


class MoFGBMLNSGAIIIMain(AbstractMoFGBMLMain):
    def __init__(self, knowledge_factory_class):
        super().__init__(MoFGBMLNSGAIIIArgs(), MoFGBMLNSGAIIIMain.run, knowledge_factory_class)

    @staticmethod
    def run(train, args, knowledge, objectives, termination, antecedent_factory, crossover):
        num_objectives_michigan = 1
        num_constraints_michigan = 0

        num_vars_pittsburgh = args.get("INITIATION_RULE_NUM")
        num_constraints_pittsburgh = 0

        rule_builder = RuleBuilderBasic(antecedent_factory,
                                        LearningBasic(train),
                                        knowledge)

        michigan_solution_builder = MichiganSolutionBuilder(num_objectives_michigan,
                                                            num_constraints_michigan,
                                                            rule_builder)

        classification = SingleWinnerRuleSelection(args.get("CACHE_SIZE"))
        classifier = Classifier(classification)

        problem = PittsburghProblem(num_vars_pittsburgh,
                                    objectives,
                                    num_constraints_pittsburgh,
                                    train,
                                    michigan_solution_builder,
                                    classifier)

        algorithm = NSGA3(ref_dirs=get_reference_directions("das-dennis", len(objectives), n_partitions=12),
                          pop_size=args.get("POPULATION_SIZE"),
                          sampling=HybridGBMLSampling(train),
                          crossover=crossover,
                          repair=PittsburghRepair(),
                          mutation=PittsburghMutation(train, knowledge),
                          save_history=True,
                          n_offsprings=args.get("OFFSPRING_POPULATION_SIZE"))

        res = minimize(problem,
                       algorithm,
                       termination=termination,
                       seed=args.get("RAND_SEED"),
                       # save_history=True,
                       verbose=True)
        return res


if __name__ == '__main__':
    runner = MoFGBMLNSGAIIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(sys.argv[1:])
