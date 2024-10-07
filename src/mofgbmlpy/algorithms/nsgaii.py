from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


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
                          sampling=self._sampling,
                          crossover=self._crossover,
                          repair=self._repair,
                          mutation=self._mutation,
                          eliminate_duplicates=False,
                          save_history=True,
                          n_offsprings=self._mofgbml_args.get("OFFSPRING_POPULATION_SIZE"))

        res = minimize(self._problem,
                       algorithm,
                       termination=self._termination,
                       seed=self._mofgbml_args.get("RAND_SEED"),
                       callback=self._callback,
                       verbose=self._verbose)
        return res