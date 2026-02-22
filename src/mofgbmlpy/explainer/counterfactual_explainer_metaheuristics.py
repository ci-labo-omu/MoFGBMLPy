from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics_abstract import (
    CounterFactualExplainerMetaheuristicsAbstract,
)
from mofgbmlpy.explainer.gbml.fuzzy_sets_sampling import FuzzySetsSampling
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_mutation import FuzzySetsMutation
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_crossover import FuzzySetsCrossover


class CounterFactualExplainerMetaheuristics(CounterFactualExplainerMetaheuristicsAbstract):
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
