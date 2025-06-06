import time

from mofgbmlpy.data.output import Output
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain


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
