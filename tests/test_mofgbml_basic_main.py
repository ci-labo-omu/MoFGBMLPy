from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.basic.mofgbml_basic_main import MoFGBMLBasicMain


def test_main():
    args = [
        "--data-name", "iris",
        "--algorithm-id", "1",
        "--experiment-id", "2",
        "--num-parallel-cores", "1",
        "--train-file", "../dataset/iris/a0_0_iris-10tra.dat",
        "--test-file", "../dataset/iris/a0_0_iris-10tst.dat",
        # "--no-plot",
        "--pretty-xml",
        "--terminate-evaluation", "1000"
    ]

    runner = MoFGBMLBasicMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(args)
    assert True
