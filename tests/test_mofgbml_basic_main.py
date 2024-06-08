from main.consts import Consts
from src.main.basic.mofgbml_basic_main import MoFGBMLBasicMain
from src.fuzzy.knowledge.knowledge import Knowledge
from simpful.fuzzy_sets import TriangleFuzzySet
from src.fuzzy.fuzzy_term.linguistic_variable_mofgbml import LinguisticVariableMoFGBML


def test_main():
    context = Knowledge.get_instance()

    fuzzy_sets = []
    for i in range(4): # nb attributes of iris
        s1 = TriangleFuzzySet(0, 0.4, 0.8, "small")
        s2 = TriangleFuzzySet(0.2, 0.6, 1, "big")
        fuzzy_sets.append(LinguisticVariableMoFGBML([s1, s2], f"x_{i}"))

    context.set_fuzzy_sets(fuzzy_sets)

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
