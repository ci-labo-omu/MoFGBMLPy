from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.basic.mofgbml_basic_main import MoFGBMLBasicMain


def test_main():
    args = [
        "iris",
        1,
        2,
        1,
        "../dataset/iris/a0_0_iris-10tra.dat",
        "../dataset/iris/a0_0_iris-10tst.dat",
    ]

    runner = MoFGBMLBasicMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(args)
    assert True
