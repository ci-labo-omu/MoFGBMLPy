from pymoo.termination import get_termination

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover
from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory

from mofgbmlpy.gbml.operator.repair.pittsburgh_repair import PittsburghRepair
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_args import MoFGBMLNSGAIIArgs
import sys

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling


class MoFGBMLNSGAIIMain(AbstractMoFGBMLMain):
    """MoFBML runner for NSGA-II"""
    def __init__(self, knowledge_factory_class):
        """Constructor
        Args:
            knowledge_factory_class (AbstractKnowledgeFactory): Knowledge factory class
        """
        super().__init__(MoFGBMLNSGAIIArgs(), knowledge_factory_class)

    def run(self):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """
        algorithm = NSGA2(pop_size=self._mofgbml_args.get("POPULATION_SIZE"),
                          sampling=HybridGBMLSampling(self._learner),
                          crossover=self._crossover,
                          repair=PittsburghRepair(),
                          mutation=PittsburghMutation(self._knowledge, self._random_gen),
                          eliminate_duplicates=False,
                          save_history=True,
                          n_offsprings=self._mofgbml_args.get("OFFSPRING_POPULATION_SIZE"))

        res = minimize(self._problem,
                       algorithm,
                       termination=self._termination,
                       seed=self._mofgbml_args.get("RAND_SEED"),
                       verbose=self._verbose)
        return res


if __name__ == '__main__':
    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(sys.argv[1:])
