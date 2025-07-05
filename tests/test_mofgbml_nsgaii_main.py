import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mofgbmlpy.data.output import Output
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain
from util import distribution_test_helper_plot_assert


def test_main_iris():
    # start = time.time()
    args = [
        "--data-name",
        "iris",
        "--algorithm-id",
        "1",
        "--experiment-id",
        "2",
        "--rand-seed",
        "2020",
        "--train-file",
        "../dataset/iris/a0_0_iris-10tra.dat",
        "--test-file",
        "../dataset/iris/a0_0_iris-10tst.dat",
        "--terminate-evaluation",
        "3000",
        "--objectives",
        "total-rule-length",
        "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        "--gen-plot",
        "--algorithm",
        "nsga2",
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
    runner.run(args)
    # results = runner.main(args)
    # elapsed = time.time() - start

    # txt = str(elapsed)+"\n"
    #
    # for objectives in results.F:
    #     txt += f"{objectives[0]},{objectives[1]}\n"

    # Output.writeln("../py_version_results/basic_main_iris_30000.txt", txt)

    assert True


def test_main_multiclass():
    args = [
        # "--data-name", "flags",
        # "--data-name", "richromatic",
        "--data-name",
        "german",
        "--algorithm-id",
        "1",
        "--experiment-id",
        "2",
        "--rand-seed",
        "2020",
        # "--train-file", "../dataset/flags/a0_0_flags-10tra.dat",
        # "--test-file", "../dataset/flags/a0_0_flags-10tst.dat",
        # "--train-file", "../dataset/richromatic/a0_0_richromatic-10tra.dat",
        # "--test-file", "../dataset/richromatic/a0_0_richromatic-10tst.dat",
        "--train-file",
        "../dataset/german/a0_0_german-10tra.dat",
        "--test-file",
        "../dataset/german/a0_0_german-10tst.dat",
        "--terminate-evaluation",
        "100",
        "--objectives",
        "total-rule-length",
        "error-rate",
        "--is-multi-label",
        "--gen-plot",
        "--verbose",
        "--algorithm",
        "nsga2",
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
    runner.run(args)

    assert True


def test_deepcopy_generations():
    args = [
        "--data-name",
        "iris",
        "--algorithm-id",
        "1",
        "--experiment-id",
        "2",
        "--rand-seed",
        "2020",
        "--train-file",
        "../dataset/iris/a0_0_iris-10tra.dat",
        "--test-file",
        "../dataset/iris/a0_0_iris-10tst.dat",
        "--terminate-generation",
        "20",
        "--objectives",
        "total-rule-length",
        "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        # "--gen-plot",
        "--algorithm",
        "nsga2",
        "--population-size",
        "10",
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
    results = runner.run(args)

    for i in range(len(results.history)):
        for j in range(i + 1, len(results.history)):
            pop1 = results.history[i].pop.get("X")
            pop2 = results.history[j].pop.get("X")
            for k in range(len(pop1)):
                for l in range(len(pop2)):
                    assert id(pop1[k][0]) != id(
                        pop2[l][0]
                    ), f"Generation {i} and {j} share a Pittsburgh solution object"

                    assert id(pop1[k][0].get_vars().base) != id(
                        pop2[l][0].get_vars().base
                    ), f"Generation {i} and {j} share a Pittsburgh solution vars object"

                    # check Michigan solutions
                    for m in range(len(pop1[k][0].get_vars())):
                        for n in range(len(pop2[l][0].get_vars())):
                            assert id(pop1[k][0].get_vars()[m]) != id(
                                pop2[l][0].get_vars()[n]
                            ), f"Generation {i} and {j} share a Michigan solution object"
                            assert id(pop1[k][0].get_vars()[m].get_vars().base) != id(
                                pop2[l][0].get_vars()[n].get_vars().base
                            ), f"Generation {i} and {j} share a Michigan solution object"
    assert True

def test_java_distribution_iris():
    num_evals = 12
    num_iters = 200
    pop_size = 5

    tests_root = Path(__file__).parent
    data_path = os.path.join(tests_root.parent, "dataset")

    args = [
        "--data-name",
        "iris",
        "--algorithm-id",
        "1",
        "--experiment-id",
        "2",
        "--train-file",
        f"{data_path}/iris/a0_0_iris-10tra.dat",
        "--test-file",
        f"{data_path}/iris/a0_0_iris-10tst.dat",
        "--terminate-evaluation",
        f"{num_evals}",
        "--objectives",
        "error-rate",
        "num-rules",
        "--population-size",
        f"{pop_size}",
        "--algorithm",
        "nsga2",
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)

    tests_data_root = os.path.join(tests_root, "test_data", "main", "nsgaii")
    data_name_config_path = os.path.join(tests_data_root, "iris")

    file_path = os.path.join(data_name_config_path, "pop.csv")
    df = pd.read_csv(file_path, header=0)

    file_path = os.path.join(data_name_config_path, "pop_rules.csv")
    df_rules = pd.read_csv(file_path, header=0)

    error_rate = []
    num_rules = []

    rule_length = []
    num_wins = []
    num_classified_patterns = []
    rule_weight = []

    old_seed = 2020

    for i in range(num_iters):
        new_seed = old_seed + i * 7

        new_args = args.copy()
        new_args.extend(["--rand-seed", f"{new_seed}"])
        res = runner.run(new_args)
        sols = res.opt.get("X")[:, 0]

        for p_sol in sols:
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
        f"Comparison on Iris Using NSGAII with a population size of {pop_size} with {num_evals} evals on {num_iters} iterations",
    )
