from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain


def test_main():
    args = [
        "--data-name", "iris",
        "--algorithm-id", "2",
        "--experiment-id", "2",
        # "--num-parallel-cores", "1",
        "--train-file", "../dataset/iris/a0_0_iris-10tra.dat",
        "--test-file", "../dataset/iris/a0_0_iris-10tst.dat",
        "--gen-plot",
        "--objectives", "num-rules", "error-rate",
        # "--terminate-evaluation", "30000",
        "--algorithm", "moead"
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
    runner.run(args)

    assert True
