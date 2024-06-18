from main.basic.mofgbml_basic_main import MoFGBMLBasicMain
from fuzzy.knowledge.knowledge import Knowledge
from simpful.fuzzy_sets import TriangleFuzzySet
from fuzzy.fuzzy_term.linguistic_variable_mofgbml import LinguisticVariableMoFGBML


def test_main():
    args = [
        "iris",
        1,
        2,
        1,
        "../dataset/iris/a0_0_iris-10tra.dat",
        "../dataset/iris/a0_0_iris-10tra.dat",
    ]

    MoFGBMLBasicMain.main(args)
    assert True
