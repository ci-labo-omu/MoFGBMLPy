from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)

from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain
import sys

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from mofgbmlpy.main.arguments.pittsburgh_style_arguments import PittsburghStyleArguments


class MoFGBMLNSGAIIMain(AbstractMoFGBMLMain):
    """MoFBML runner for NSGA-II"""

    def __init__(self, knowledge_factory_class):
        """Constructor

        Args:
            knowledge_factory_class (AbstractKnowledgeFactory): Knowledge factory class
        """
        super().__init__(PittsburghStyleArguments("nsga2"), knowledge_factory_class)

    def run(self):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """
        algorithm = NSGA2(
            pop_size=self._mofgbml_args.get("POPULATION_SIZE"),
            sampling=self._sampling,
            crossover=self._crossover,
            repair=self._repair,
            mutation=self._mutation,
            eliminate_duplicates=False,
            save_history=True,
            n_offsprings=self._mofgbml_args.get("OFFSPRING_POPULATION_SIZE"),
        )

        res = minimize(
            self._problem,
            algorithm,
            termination=self._termination,
            seed=self._mofgbml_args.get("RAND_SEED"),
            callback=self._callback,
            verbose=self._verbose,
        )
        return res


if __name__ == "__main__":
    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(sys.argv[1:])
