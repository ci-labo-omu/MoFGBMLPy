from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics_abstract import (
    CounterFactualExplainerMetaheuristicsAbstract,
)
from mofgbmlpy.explainer.gbml.fuzzy_sets_sampling_fk import FuzzySetsSamplingFK
import numpy as np
from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation

from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring_michigan import (
    UniformCrossoverSingleOffspringMichigan,
)

from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder


class CounterFactualExplainerMetaheuristicsFK(CounterFactualExplainerMetaheuristicsAbstract):
    """Counterfactual explainer using metaheuristics to find CF rules using fixed knowledge base"""

    # fixed knowledge
    def __init__(
        self,
        classifier,
        changed_rule_index,
        target_class,
        test_set,
        mutation_prob=0.7,
        crossover_prob=0.7,
        use_search_space_crowding=False,
        objectives=["confidence_loss", "change_loss"],
        n_gen=60,
        pop_size=60,
    ):
        """Constructor

        Args:
            classifier (Classifier): The classifier for which to find counterfactual explanations, used to get the knowledge for the mutation operator
            changed_rule_index (int): The index of the rule to change in the counterfactual explanation, used to get the initial rule for the sampling and to apply the changes in the mutation and crossover operators
            target_class (int): The target class for the counterfactual explanation, used to calculate the confidence loss objective
            test_set (DataSet): The test set to evaluate the solutions on, used to calculate the confidence loss objective
            mutation_prob (float, optional): The probability of applying mutation. Defaults to 0.7.
            crossover_prob (float, optional): The probability of applying crossover. Defaults to 0.7.
            use_search_space_crowding (bool, optional): Whether to use search space crowding instead of objective space crowding. Defaults to False.
            objectives (list of str, optional): The list of objectives to optimize. Defaults to ["confidence_loss", "change_loss"].
            n_gen (int, optional): The number of generations for the optimization. Defaults to 60.
            pop_size (int, optional): The population size for the optimization. Defaults to 60.
        """
        knowledge = classifier.get_var(0).get_rule().get_knowledge()
        random_gen = np.random.Generator(np.random.MT19937(seed=2022))
        rule_builder = classifier.get_var(0).get_rule_builder()

        michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)

        sampling = FuzzySetsSamplingFK(michigan_solution_builder)
        mutation = MichiganMutation(knowledge, mutation_prob, random_gen)
        crossover = UniformCrossoverSingleOffspringMichigan(random_gen, crossover_prob)

        n_gen = n_gen
        pop_size = pop_size

        super().__init__(
            classifier,
            changed_rule_index,
            target_class,
            test_set,
            sampling,
            mutation,
            crossover,
            use_search_space_crowding,
            objectives,
            n_gen,
            pop_size,
        )
