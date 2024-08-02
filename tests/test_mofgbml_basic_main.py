import time

from mofgbmlpy.data.output import Output
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain


def test_main():
    start = time.time()
    # args = [
    #     "--data-name", "pima",
    #     # "--data-name", "iris",
    #     "--algorithm-id", "1",
    #     "--experiment-id", "2",
    #     # "--num-parallel-cores", "1",
    #     "--train-file", "../dataset/pima/a0_0_pima-10tra.dat",
    #     "--test-file", "../dataset/pima/a0_0_pima-10tst.dat",
    #     # "--train-file", "../dataset/iris/a0_0_iris-10tra.dat",
    #     # "--test-file", "../dataset/iris/a0_0_iris-10tst.dat",
    #     # "--pretty-xml",
    #     # "--terminate-evaluation", "30000",
    #     "--terminate-generation", "100",
    #     "--objectives", "num-rules", "error-rate"
    # ]

    args = [
        "--data-name", "iris",
        "--algorithm-id", "1",
        "--experiment-id", "2",
        "--train-file", "../dataset/iris/a0_0_iris-10tra.dat",
        "--test-file", "../dataset/iris/a0_0_iris-10tst.dat",
        "--terminate-generation", "50",
        "--objectives", "num-rules", "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        "--no-plot"
    ]

    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    results = runner.main(args)
    elapsed = time.time() - start

    txt = str(elapsed)+"\n"

    for objectives in results.F:
        txt += f"{objectives[0]},{objectives[1]}\n"

    Output.writeln("../py_version_results/basic_main_iris_30000.txt", txt)

    assert True
