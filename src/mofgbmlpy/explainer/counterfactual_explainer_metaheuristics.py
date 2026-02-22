from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics_abstract import (
    CounterFactualExplainerMetaheuristicsAbstract,
)
from mofgbmlpy.explainer.gbml.fuzzy_sets_sampling import FuzzySetsSampling
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_mutation import FuzzySetsMutation
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_crossover import FuzzySetsCrossover


class CounterFactualExplainerMetaheuristics(CounterFactualExplainerMetaheuristicsAbstract):
    """Counterfactual explainer using metaheuristics to find CF rules using non-fixed knowledge base"""

    def __init__(
        self,
        classifier,
        changed_rule_index,
        target_class,
        test_set,
        mutation_fs_type_prob=0.5,
        sampling_noise_str=0.1,
        mutation_prob=0.7,
        mutated_param_prob=0.6,
        mutation_revert_to_initial_prob=0.0,
        crossover_prob=0.7,
        crossover_p1_selected_prob=0.5,
        sampling_fs_type_prob=0.0,
        sampling_change_fs_params_prob=1.0,
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
            mutation_fs_type_prob (float, optional): The probability of changing the type of a fuzzy set to DontCareFuzzySet in the mutation operator. Defaults to 0.5.
            sampling_noise_str (float, optional): The standard deviation of the noise added to the fuzzy sets parameters in the sampling operator. Defaults to 0.1.
            mutated_param_prob (float, optional): The probability of changing the parameters of a fuzzy set by adding noise in the mutation operator. Defaults to 0.6.
            mutation_revert_to_initial_prob (float, optional): The probability of reverting a fuzzy set to its initial parameters in the mutation operator. Defaults to 0.0.
            crossover_prob (float, optional): The probability of applying crossover. Defaults to 0.7.
            crossover_p1_selected_prob (float, optional): The probability of selecting a parent from the first half of the population in the crossover operator. Defaults to 0.5.
            sampling_fs_type_prob (float, optional): The probability of changing the type of a fuzzy set to DontCareFuzzySet in the sampling operator. Defaults to 0.0.
            sampling_change_fs_params_prob (float, optional): The probability of changing the parameters of a fuzzy set by adding noise in the sampling operator. Defaults to 1.0.
            use_search_space_crowding (bool, optional): Whether to use search space crowding instead of objective space crowding. Defaults to False.
            objectives (list of str, optional): The list of objectives to optimize. Defaults to ["confidence_loss", "change_loss"].
            n_gen (int, optional): The number of generations for the optimization. Defaults to 60.
            pop_size (int, optional): The population size for the optimization. Defaults to 60.
        """
        sampling = FuzzySetsSampling(sampling_noise_str, sampling_fs_type_prob, sampling_change_fs_params_prob)
        mutation = FuzzySetsMutation(
            mutation_prob, mutated_param_prob, mutation_fs_type_prob, mutation_revert_to_initial_prob
        )
        crossover = FuzzySetsCrossover(crossover_prob, crossover_p1_selected_prob)

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
