import json
from scipy.stats import ttest_ind
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from pymoo.core.population import Population
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.objectives.pittsburgh.error_rate import ErrorRate
from mofgbmlpy.gbml.objectives.pittsburgh.num_rules import NumRules
import pandas as pd
from matplotlib import pyplot as plt

import csv
import os

import numpy as np

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_multi import LearningMulti
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.fuzzy.rule.rule_builder_multi import RuleBuilderMulti
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution

from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover

from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover

from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover
from mofgbmlpy.main.arguments.arguments import Arguments


def get_datasets(datasets_dir="../dataset"):
    datasets = {}
    for folder in os.listdir(datasets_dir):
        datasets[folder] = []
        for items in os.walk(os.path.join(datasets_dir, folder)):
            if "subdata" in items[0].split(os.sep):
                continue
            files = []
            for file in items[2]:
                path = os.path.join(items[0], file)
                try:
                    with open(path, newline="") as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        if len(header) < 3:
                            raise Exception("Invalid header")
                        for i in range(3):
                            int(header[i])
                    files.append(path)
                except:
                    pass  # Invalid format (not csv or invalid header)

            datasets[folder] += files
    return datasets


def get_a0_0_iris_train_test():
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = Arguments()
    args.set("TRAIN_FILE", f"{root_folder}/dataset/iris/a0_0_iris-10tra.dat")
    args.set("TEST_FILE", f"{root_folder}/dataset/iris/a0_0_iris-10tst.dat")
    args.set("IS_MULTI_LABEL", False)

    return Input.get_train_test_files(args)


def get_a0_0_pima_train_test():
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = Arguments()
    args.set("TRAIN_FILE", f"{root_folder}/dataset/pima/a0_0_pima-10tra.dat")
    args.set("TEST_FILE", f"{root_folder}/dataset/pima/a0_0_pima-10tst.dat")
    args.set("IS_MULTI_LABEL", False)

    return Input.get_train_test_files(args)


def get_a0_0_german_train_test():
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = Arguments()
    args.set("TRAIN_FILE", f"{root_folder}/dataset/german/a0_0_german-10tra.dat")
    args.set("TEST_FILE", f"{root_folder}/dataset/german/a0_0_german-10tst.dat")
    args.set("IS_MULTI_LABEL", True)

    return Input.get_train_test_files(args)


def create_michigan_sol(training_data_set, seed=2022, antecedent_indices=None, consequent=None, is_multi_label=False):
    random_gen = np.random.Generator(np.random.MT19937(seed))

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()
    antecedent_factory = HeuristicAntecedentFactory(training_data_set, knowledge, False, 0.7, 5, random_gen)

    if is_multi_label:
        consequent_factory = LearningMulti(training_data_set)
        rule_builder = RuleBuilderMulti(antecedent_factory, consequent_factory, knowledge)
    else:
        consequent_factory = LearningBasic(training_data_set)
        rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)

    solution = MichiganSolution(random_gen, 2, 0, rule_builder)

    if antecedent_indices is not None:
        solution.set_vars(antecedent_indices)
        solution.learning()

    if consequent is not None:
        solution.get_rule().set_consequent(consequent)

    return solution


def create_pittsburgh_sol(training_data_set, classification, michigan_sols=None, michigan_solution_builder=None):
    if michigan_sols is None:
        michigan_sols = [create_michigan_sol(training_data_set)]

    sol = PittsburghSolution(len(michigan_sols), 2, 0, classification, michigan_solution_builder=michigan_solution_builder, do_init_vars=False)
    sol.set_vars(michigan_sols)
    return sol

def float_eq(value1, value2, precision=1e-6):
    return abs(value1 - value2) < precision


def crossover_test_helper_run(crossover, problem, pop, parents, data_name, data_name_config_path):
    file_path = os.path.join(data_name_config_path, "offsprings.csv")
    df = pd.read_csv(file_path, header=0)

    file_path = os.path.join(data_name_config_path, "offsprings_rules.csv")
    df_rules = pd.read_csv(file_path, header=0)

    num_iters = len(df)
    error_rate = np.zeros(num_iters)
    total_rule_length = np.zeros(num_iters)

    rule_weight = []
    rule_length = []
    num_wins = []
    num_classified_patterns = []

    for i in range(num_iters):
        offspring = crossover.do(problem, pop, parents=parents)
        problem.evaluate(offspring.get("X"))

        child = offspring[0].X[0]
        error_rate[i] = child.get_objective(0)
        total_rule_length[i] = child.get_objective(1)

        for rule in child.get_vars():
            rule_weight.append(rule.get_rule_weight_py().get_value())
            rule_length.append(rule.get_length())
            num_wins.append(rule.get_num_wins())
            num_classified_patterns.append(rule.get_fitness())

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Pittsburgh Solutions Comparison on {data_name} using {crossover.__class__.__name__} on {num_iters} iterations", fontweight="bold")

    axs[0].hist(df["error_rate"], bins=50, alpha=0.5, label="Java", color="blue")
    axs[0].hist(error_rate, bins=50, alpha=0.5, label="Python", color="orange")
    axs[0].set_title(f"Error Rate Distribution ({data_name})")
    axs[0].set_xlabel("Error Rate")
    axs[0].set_ylabel("Frequency")
    axs[0].set_xlim(0, 1)
    axs[0].legend()

    max_rule_length = max(df["total_rule_length"].max(), total_rule_length.max())
    x = np.arange(0, max_rule_length + 1)
    java_counts = df["total_rule_length"].value_counts().reindex(x, fill_value=0)
    python_counts = pd.Series(total_rule_length).value_counts().reindex(x, fill_value=0)
    axs[1].bar(x - 0.2, java_counts, width=0.4, label="Java", color="blue", alpha=0.5)
    axs[1].bar(x + 0.2, python_counts, width=0.4, label="Python", color="orange", alpha=0.5)
    axs[1].set_title(f"Total Rule Length Distribution ({data_name})")
    axs[1].set_xlabel("Total Rule Length")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlim(0, max_rule_length + 1)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # now compare rules, plot each vars on one row (2 plots) similarly
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    fig.suptitle(f"Michigan Solutions Comparison on {data_name} using {crossover.__class__.__name__} on {num_iters} iterations", fontweight="bold")
    axs[0].hist(df_rules["rule_weight"], bins=50, alpha=0.5, label="Java", color="blue")
    axs[0].hist(rule_weight, bins=50, alpha=0.5, label="Python", color="orange")
    axs[0].set_title(f"Rule Weight Distribution ({data_name})")
    axs[0].set_xlabel("Rule Weight")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    max_rule_length = max(df_rules["rule_length"].max(), max(rule_length))
    x = np.arange(0, max_rule_length + 1)
    java_counts = df_rules["rule_length"].value_counts().reindex(x, fill_value=0)
    python_counts = pd.Series(rule_length).value_counts().reindex(x, fill_value=0)
    axs[1].bar(x - 0.2, java_counts, width=0.4, label="Java", color="blue", alpha=0.5)
    axs[1].bar(x + 0.2, python_counts, width=0.4, label="Python", color="orange", alpha=0.5)
    axs[1].set_title(f"Rule Length Distribution ({data_name})")
    axs[1].set_xlabel("Rule Length")
    axs[1].set_ylabel("Frequency")
    axs[1].legend()
    axs[1].set_xlim(0, max_rule_length + 1)

    axs[2].hist(df_rules["num_wins"], bins=50, alpha=0.5, label="Java", color="blue")
    axs[2].hist(num_wins, bins=50, alpha=0.5, label="Python", color="orange")
    axs[2].set_title(f"Number of Wins Distribution ({data_name})")
    axs[2].set_xlabel("Number of Wins")
    axs[2].set_ylabel("Frequency")
    axs[2].legend()

    axs[3].hist(df_rules["num_classified_patterns"], bins=50, alpha=0.5, label="Java", color="blue")
    axs[3].hist(num_classified_patterns, bins=50, alpha=0.5, label="Python", color="orange")
    axs[3].set_title(f"Number of Classified Patterns Distribution ({data_name})")
    axs[3].set_xlabel("Number of Classified Patterns")
    axs[3].set_ylabel("Frequency")
    axs[3].legend()

    plt.tight_layout()
    plt.show()

    # statistical test to compare distributions

    t_stat, p_value = ttest_ind(df["error_rate"], error_rate)
    assert p_value > 0.05, f"Error rate distributions are significantly different (p-value: {p_value})"

    t_stat, p_value = ttest_ind(df["total_rule_length"], total_rule_length)
    assert p_value > 0.05, f"Total rule length distributions are significantly different (p-value: {p_value})"

    t_stat, p_value = ttest_ind(df_rules["rule_weight"], rule_weight)
    assert p_value > 0.05, f"Rule weight distributions are significantly different (p-value: {p_value})"

    t_stat, p_value = ttest_ind(df_rules["rule_length"], rule_length)
    assert p_value > 0.05, f"Rule length distributions are significantly different (p-value: {p_value})"

    t_stat, p_value = ttest_ind(df_rules["num_wins"], num_wins)
    assert p_value > 0.05, f"Number of wins distributions are significantly different (p-value: {p_value})"

    t_stat, p_value = ttest_ind(df_rules["num_classified_patterns"], num_classified_patterns)
    assert (
            p_value > 0.05
    ), f"Number of classified patterns distributions are significantly different (p-value: {p_value})"


def crossover_test_helper_init_config(tests_data_root, data_name):
    data_name_config_path = os.path.join(tests_data_root, data_name)
    if data_name == "iris":
        train, _ = get_a0_0_iris_train_test()
    elif data_name == "pima":
        train, _ = get_a0_0_pima_train_test()
    else:
        raise ValueError(f"Unknown data name: {data_name}")

    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent_factory = HeuristicAntecedentFactory(train, knowledge, False, 0.8, 5, random_gen)
    consequent_factory = LearningBasic(train)
    rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)

    classification = SingleWinnerRuleSelection()
    objectives = np.array([ErrorRate(train), NumRules()])
    michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)

    problem = PittsburghProblem(train.get_num_dim(), objectives, 0, train, michigan_solution_builder, classification)

    indices = json.load(open(os.path.join(data_name_config_path, "parents.json"), "r"))

    sol1_indices = np.array(indices[0], dtype=np.int32)
    sol2_indices = None
    if len(indices) > 1:
        sol2_indices = np.array(indices[1], dtype=np.int32)

    michigan_sols = np.empty(len(sol1_indices), dtype=object)
    for i, indices in enumerate(sol1_indices):
        michigan_sols[i] = create_michigan_sol(train, antecedent_indices=indices)

    sol1 = create_pittsburgh_sol(train, classification, michigan_sols, michigan_solution_builder)

    sol2 = None
    if sol2_indices is not None:
        michigan_sols = np.empty(len(sol2_indices), dtype=object)
        for i, indices in enumerate(sol2_indices):
            michigan_sols[i] = create_michigan_sol(train, antecedent_indices=indices)

        sol2 = create_pittsburgh_sol(train, classification, michigan_sols, michigan_solution_builder)

    if sol2 is None:
        pop = Population.new(X=np.array([[sol1]], dtype=object))
    else:
        pop = Population.new(X=np.array([[sol1], [sol2]], dtype=object))

    return pop, problem, data_name_config_path, random_gen


def get_hybrid_crossover(problem, random_gen, min_num_rules=1, max_num_rules=60,
                         pittsburgh_crossover_probability=0.9,
                         michigan_crossover_probability=0.9, crossover_probability=1, rule_change_rate=0.2):

    pittsburgh_crossover = PittsburghCrossover(min_num_rules, max_num_rules, random_gen,
                                               pittsburgh_crossover_probability)

    michigan_crossover = MichiganCrossover(
        rule_change_rate,
        problem.get_training_set(),
        problem.get_knowledge(),
        max_num_rules,
        random_gen,
        michigan_crossover_probability
    )

    crossover = HybridGBMLCrossover(
        random_gen,
        michigan_crossover_probability,
        michigan_crossover,
        pittsburgh_crossover,
        crossover_probability
    )

    return crossover
