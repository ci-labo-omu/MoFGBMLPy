from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.moead.mofgbml_moead_main import MoFGBMLMOEADMain


def test_main():
    args = [
        "--data-name", "iris",
        "--algorithm-id", "1",
        "--experiment-id", "2",
        "--num-parallel-cores", "1",
        "--train-file", "../dataset/iris/a0_0_iris-10tra.dat",
        "--test-file", "../dataset/iris/a0_0_iris-10tst.dat",
        # "--no-plot",
        "--objectives", "num-rules", "error-rate"
    ]

    runner = MoFGBMLMOEADMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(args)
    assert True
