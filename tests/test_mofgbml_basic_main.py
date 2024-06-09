from main.consts import Consts
from src.main.basic.mofgbml_basic_main import MoFGBMLBasicMain
from src.fuzzy.knowledge.knowledge import Knowledge
from simpful.fuzzy_sets import TriangleFuzzySet
from src.fuzzy.fuzzy_term.linguistic_variable_mofgbml import LinguisticVariableMoFGBML


def test_main():
    args = [
        "__data_name",
        1,
        2,
        1,
        "../dataset/iris/a0_0_iris-10tra.dat",
        "../dataset/iris/a0_0_iris-10tra.dat",
    ]

    MoFGBMLBasicMain.main(args)
    assert True
