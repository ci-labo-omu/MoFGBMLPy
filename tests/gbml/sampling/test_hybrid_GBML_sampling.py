import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.objectives.pittsburgh.error_rate import ErrorRate
from mofgbmlpy.gbml.objectives.pittsburgh.num_rules import NumRules
from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder

from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution
from mofgbmlpy.main.arguments.arguments import Arguments
from util import get_a0_0_iris_train_test, helper_init_config, distribution_test_helper_plot_assert
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling


def test_sampling_example():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    train, _ = get_a0_0_iris_train_test()
    pop_size = 10

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent_factory = HeuristicAntecedentFactory(train, knowledge, False, 0.8, 5, random_gen)
    consequent_factory = LearningBasic(train)
    rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)
    objectives = np.array([ErrorRate(train), NumRules()])
    michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)
    classification = SingleWinnerRuleSelection()

    sampling = HybridGBMLSampling(consequent_factory)
    problem = PittsburghProblem(30, objectives, 0, train, michigan_solution_builder, classification)
    pop = sampling._do(problem, pop_size)

    assert len(pop) == pop_size
    assert len(pop[0]) == 1
    assert isinstance(pop[0][0], PittsburghSolution)


def test_java_distribution():
    tests_root = Path(__file__).parents[2]
    tests_data_root = os.path.join(tests_root, "test_data", "sampling", "hybrid_GBML_sampling")

    data_names = [name for name in os.listdir(tests_data_root) if os.path.isdir(os.path.join(tests_data_root, name))]

    pop_size = 60

    for data_name in data_names:
        data_name_config_path = os.path.join(tests_data_root, data_name)

        file_path = os.path.join(data_name_config_path, "offsprings.csv")
        df = pd.read_csv(file_path, header=0)

        file_path = os.path.join(data_name_config_path, "offsprings_rules.csv")
        df_rules = pd.read_csv(file_path, header=0)

        pop, problem, random_gen = helper_init_config(data_name)
        train = problem.get_training_set()

        consequent_factory = LearningBasic(train)

        sampling = HybridGBMLSampling(consequent_factory)

        error_rate = []
        num_rules = []

        rule_weight = []
        rule_length = []
        num_wins = []
        num_classified_patterns = []

        _, problem, random_gen = helper_init_config(data_name)

        num_iters = len(df) // pop_size

        for _ in range(num_iters):
            pop = sampling.do(problem, pop_size)
            problem.remove_no_winner_michigan_solution(pop.get("X"))
            problem.evaluate(pop.get("X"))

            for ind in pop:
                p_sol = ind.X[0]
                error_rate.append(p_sol.get_error_rate())
                num_rules.append(p_sol.get_num_vars())

                assert p_sol.get_error_rate() == p_sol.get_objective(0)
                assert p_sol.get_num_vars() == p_sol.get_objective(1)

                for m_sol in p_sol.get_vars():
                    rule_length.append(m_sol.get_rule().get_length())
                    num_wins.append(m_sol.get_num_wins())
                    num_classified_patterns.append(m_sol.get_fitness())
                    rule_weight.append(m_sol.get_rule_weight_py().get_value())

        distribution_test_helper_plot_assert(
            error_rate,
            num_rules,
            rule_weight,
            rule_length,
            num_wins,
            num_classified_patterns,
            df,
            df_rules,
            f"Comparison on {data_name} Using HybridGBMLSampling with a population size of {pop_size} on {num_iters} iterations",
        )
