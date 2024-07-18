from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.basic.mofgbml_basic_main import MoFGBMLBasicMain


def test_main():
    args = [
        # "--data-name", "iris",
        "--data-name", "german",
        "--algorithm-id", "1",
        "--experiment-id", "2",
        "--num-parallel-cores", "1",
        "--train-file", "../dataset/iris/a0_0_iris-10tra.dat",
        "--test-file", "../dataset/iris/a0_0_iris-10tst.dat",
        # "--train-file", "../dataset/yeast_multi/yeast-10-1tra.dat",
        # "--test-file", "../dataset/yeast_multi/yeast-10-1tst.dat",
        # "--is-multi-label",
        # "--no-plot",
        "--pretty-xml",
        "--terminate-evaluation", "1000",
        "--objectives", "num-rules", "error-rate"
    ]

    runner = MoFGBMLBasicMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(args)
    assert True
