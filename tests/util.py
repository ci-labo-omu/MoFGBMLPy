from scipy.stats import wasserstein_distance
import json
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

    sol = PittsburghSolution(
        len(michigan_sols),
        2,
        0,
        classification,
        michigan_solution_builder=michigan_solution_builder,
        do_init_vars=False,
    )
    sol.set_vars(michigan_sols)
    return sol


def float_eq(value1, value2, precision=1e-6):
    return abs(value1 - value2) < precision  # TODO: change to use pytest.approx, or another function instead


def compare_distribution(x, y, var_name, relative_tol=0.01):
    combined = np.concatenate([x, y])
    data_range = np.max(combined) - np.min(combined)

    if data_range == 0:
        return

    threshold = relative_tol * data_range
    w_dist = wasserstein_distance(x, y)

    assert (
        w_dist < threshold
    ), f"{var_name} distributions are too different (Wasserstein distance: {w_dist} >= {threshold})"

def plot_comparison_plot(ax, python_data, java_data, var_name, x_lim=None, use_bars=False):
    max_val = max(np.max(java_data), np.max(python_data))
    min_val = min(np.min(java_data), np.min(python_data))

    if use_bars:
        x = np.arange(0, max_val + 1)
        space = max_val / 10

        java_counts = pd.Series(java_data).value_counts().reindex(x, fill_value=0)
        python_counts = pd.Series(python_data).value_counts().reindex(x, fill_value=0)

        ax.bar(x - 0.2, java_counts, width=0.4, label="Java", color="blue", alpha=0.5)
        ax.bar(x + 0.2, python_counts, width=0.4, label="Python", color="orange", alpha=0.5)
        ax.set_xlim((-2 * space, max_val + 2 * space))
    else:
        space = (max_val - min_val) / 10
        ax.hist(java_data, bins=50, alpha=0.5, label="Java", color="blue")
        ax.hist(python_data, bins=50, alpha=0.5, label="Python", color="orange")
        ax.set_xlim((-space, max_val + space))

    if x_lim is not None:
        ax.set_xlim(x_lim)

    ax.set_title(f"{var_name} Distribution")
    ax.set_xlabel(var_name),
    ax.set_ylabel("Frequency")
    ax.legend()

    return ax


def crossover_test_helper_plot_assert(error_rate, num_rules, rule_weight, rule_length,
                                      num_wins, num_classified_patterns,
                                      df, df_rules, title):
    # fix imprecision issues
    precision = 6  # 1e-6
    error_rate = np.round(error_rate, precision)
    num_rules = np.round(num_rules, precision)
    rule_weight = np.round(rule_weight, precision)
    rule_length = np.round(rule_length, precision)
    num_wins = np.round(num_wins, precision)
    num_classified_patterns = np.round(num_classified_patterns, precision)

    java_error_rate = np.round(df["error_rate"].values, precision)
    java_num_rules = np.round(df["num_rules"].values, precision)
    java_rule_weight = np.round(df_rules["rule_weight"].values, precision)
    java_rule_length = np.round(df_rules["rule_length"].values, precision)
    java_num_wins = np.round(df_rules["num_wins"].values, precision)
    java_num_classified_patterns = np.round(df_rules["num_classified_patterns"].values, precision)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"(Pittsburgh Solutions) {title}",
        fontweight="bold",
    )

    axs[0] = plot_comparison_plot(axs[0], error_rate, java_error_rate, "Error Rate")

    axs[1] = plot_comparison_plot(axs[1], num_rules, java_num_rules, f"Number of Rules",
                                  use_bars=True)

    plt.tight_layout()
    plt.show()

    # now we compare rules, we plot each vars on one row (2 plots) similarly
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    fig.suptitle(
        f"(Michigan Solutions) {title}",
        fontweight="bold",
    )

    axs[0] = plot_comparison_plot(axs[0], rule_weight, java_rule_weight, "Rule Weight")

    axs[1] = plot_comparison_plot(axs[1], rule_length, java_rule_length, f"Rule Length", use_bars=True)

    axs[2] = plot_comparison_plot(axs[2], num_wins, java_num_wins, f"Number of Wins")

    axs[3] = plot_comparison_plot(axs[3], num_classified_patterns, java_num_classified_patterns,
                                  f"Number of Classified Patterns")

    plt.tight_layout()
    plt.show()

    compare_distribution(java_error_rate, error_rate, "Error rate")
    compare_distribution(java_num_rules, num_rules, "Number of rules")
    compare_distribution(java_rule_weight, rule_weight, "Rule weight")
    compare_distribution(java_rule_length, rule_length, "Rule length")
    compare_distribution(java_num_wins, num_wins, "Number of wins")
    compare_distribution(java_num_classified_patterns, num_classified_patterns, "Number of classified patterns")


def crossover_test_helper_run(crossover, problem, pop, parents, data_name, data_name_config_path):
    file_path = os.path.join(data_name_config_path, "offsprings.csv")
    df = pd.read_csv(file_path, header=0)

    file_path = os.path.join(data_name_config_path, "offsprings_rules.csv")
    df_rules = pd.read_csv(file_path, header=0)

    num_iters = len(df)
    error_rate = np.zeros(num_iters)
    num_rules = np.zeros(num_iters)

    rule_weight = []
    rule_length = []
    num_wins = []
    num_classified_patterns = []

    for i in range(num_iters):
        offspring = crossover.do(problem, pop, parents=parents)
        problem.evaluate(offspring.get("X"))

        child = offspring[0].X[0]
        error_rate[i] = child.get_error_rate()
        num_rules[i] = child.get_num_vars()

        assert child.get_error_rate() == child.get_objective(0)
        assert child.get_num_vars() == child.get_objective(1)

        for rule in child.get_vars():
            rule_weight.append(rule.get_rule_weight_py().get_value())
            rule_length.append(rule.get_length())
            num_wins.append(rule.get_num_wins())
            num_classified_patterns.append(rule.get_fitness())

    crossover_test_helper_plot_assert(
        error_rate, num_rules, rule_weight, rule_length,
        num_wins, num_classified_patterns, df, df_rules,
        title=f"Comparison on {data_name} using {crossover.__class__.__name__} on {num_iters} iterations"
    )


def crossover_test_helper_init_config(tests_data_root, data_name):
    # set seed of pymoo
    np.random.seed(2022)
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


def get_hybrid_crossover(
    problem,
    random_gen,
    min_num_rules=1,
    max_num_rules=60,
    pittsburgh_crossover_probability=0.9,
    michigan_crossover_probability=0.9,
    michigan_ope_probability=0.5,
    crossover_probability=1,
    rule_change_rate=0.2,
):

    pittsburgh_crossover = PittsburghCrossover(
        min_num_rules, max_num_rules, random_gen, pittsburgh_crossover_probability
    )

    michigan_crossover = MichiganCrossover(
        rule_change_rate,
        problem.get_training_set(),
        problem.get_knowledge(),
        max_num_rules,
        random_gen,
        michigan_crossover_probability,
    )

    crossover = HybridGBMLCrossover(
        random_gen, michigan_ope_probability, michigan_crossover, pittsburgh_crossover, crossover_probability
    )

    return crossover
