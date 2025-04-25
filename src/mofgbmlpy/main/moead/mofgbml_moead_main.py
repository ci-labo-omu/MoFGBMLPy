from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain
import sys

from pymoo.optimize import minimize

from mofgbmlpy.main.arguments.pittsburgh_style_arguments import PittsburghStyleArguments


class MoFGBMLMOEADMain(AbstractMoFGBMLMain):
    """MoFBML runner for MOEAD"""

    def __init__(self, knowledge_factory_class):
        """Constructor

        Args:
            knowledge_factory_class (AbstractKnowledgeFactory): Knowledge factory class
        """
        super().__init__(PittsburghStyleArguments("moead"), knowledge_factory_class)

    def run(self):
        """Run MoFGBML

        Returns:
            pymoo.core.result.Result: Result of the run
        """

        ref_dirs = get_reference_directions(
            "uniform", self._problem.get_num_objectives(), n_partitions=self._mofgbml_args.get("POPULATION_SIZE") - 1
        )  # TODO: works for 2 objectives, but change it for 1 or 3 and more objectives

        # Note: if num_obj <=2, gbml uses Tschebyscheff
        algorithm = MOEAD(
            ref_dirs,
            n_neighbors=self._mofgbml_args.get("NEIGHBORHOOD_SIZE"),
            prob_neighbor_mating=self._mofgbml_args.get("NEIGHBORHOOD_SELECTION_PROBABILITY"),
            sampling=self._sampling,
            crossover=self._crossover,
            repair=self._repair,
            mutation=self._mutation,
        )

        res = minimize(
            self._problem,
            algorithm,
            self._termination,
            seed=self._mofgbml_args.get("RAND_SEED"),
            verbose=self._verbose,
            callback=self._callback,
            save_history=True,
        )

        return res


if __name__ == "__main__":
    runner = MoFGBMLMOEADMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(sys.argv[1:])
