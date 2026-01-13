import random
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.optimize import minimize
from tqdm import tqdm

from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics_abstract import CounterFactualExplainerMetaheuristicsAbstract
from mofgbmlpy.explainer.gbml.crowding_function_x import CrowdingFunctionX
from mofgbmlpy.explainer.gbml.fuzzy_sets_sampling_fk import FuzzySetsSamplingFK
from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.explainer.gbml.fuzzy_sets_sampling import FuzzySetsSampling
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_mutation import FuzzySetsMutation
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_crossover import FuzzySetsCrossover
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
import os
import numpy as np
from mofgbmlpy.explainer.gbml.fuzzy_sets_eliminate_duplicates import FuzzySetsEliminateDuplicates
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_survival import FuzzySetsSurvival
from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover

from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation

from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling

from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover

from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover

from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation

from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring_michigan import \
    UniformCrossoverSingleOffspringMichigan
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain
import pandas as pd


class CounterFactualExplainerMetaheuristicsFK(CounterFactualExplainerMetaheuristicsAbstract):
    # fixed knowledge
    def __init__(
        self,
        classifier,
        changed_rule_index,
        target_class,
        test_set,
        mutation_prob=0.7,
        crossover_prob=0.7,
        sampling_change_prob=0.2,
        use_search_space_crowding=False,
        objectives=["confidence_loss", "change_loss"],
        n_gen=60,
        pop_size=60,
    ):
        knowledge = classifier.get_var(0).get_rule().get_knowledge()
        random_gen = np.random.Generator(np.random.MT19937(seed=2022))

        sampling = FuzzySetsSamplingFK(knowledge, sampling_change_prob)
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
            pop_size
        )
