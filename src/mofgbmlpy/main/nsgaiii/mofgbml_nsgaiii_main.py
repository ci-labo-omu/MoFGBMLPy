from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain

import sys

from pymoo.optimize import minimize


from mofgbmlpy.main.arguments.pittsburgh_style_arguments import PittsburghStyleArguments


class MoFGBMLNSGAIIIMain(AbstractMoFGBMLMain):
    """MoFBML runner for NSGA-III"""

    def __init__(self, knowledge_factory_class):
        """Constructor

        Args:
            knowledge_factory_class (AbstractKnowledgeFactory): Knowledge factory class
        """
        super().__init__(PittsburghStyleArguments("nsga3"), knowledge_factory_class)

    def run(self):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """
        algorithm = NSGA3(
            ref_dirs=get_reference_directions("das-dennis", len(self._objectives), n_partitions=12),
            pop_size=self._mofgbml_args.get("POPULATION_SIZE"),
            sampling=self._sampling,
            crossover=self._crossover,
            repair=self._repair,
            mutation=self._mutation,
            n_offsprings=self._mofgbml_args.get("OFFSPRING_POPULATION_SIZE"),
        )

        res = minimize(
            self._problem,
            algorithm,
            termination=self._termination,
            seed=self._mofgbml_args.get("RAND_SEED"),
            save_history=True,
            callback=self._callback,
            verbose=self._verbose,
        )
        return res


if __name__ == "__main__":
    runner = MoFGBMLNSGAIIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(sys.argv[1:])
