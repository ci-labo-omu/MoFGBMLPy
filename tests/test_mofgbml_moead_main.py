from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.moead.mofgbml_moead_main import MoFGBMLMOEADMain


def test_main():
    args = [
        "iris",
        1,
        2,
        1,
        "../dataset/iris/a0_0_iris-10tra.dat",
        "../dataset/iris/a0_0_iris-10tst.dat",
    ]

    runner = MoFGBMLMOEADMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(args)
    assert True
