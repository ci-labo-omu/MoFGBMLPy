from src.main.basic.mofgbml_basic_main import MoFGBMLBasicMain
from src.fuzzy.knowledge.knowledge import Context
from src.fuzzy.fuzzy_term.fuzzy_term_triangular import FuzzyTermTriangular
def test_main():
    context = Context.get_instance()

    fuzzy_sets = []
    for i in range(103): # nb attributes
        fuzzy_sets.append([
            FuzzyTermTriangular(0, 0.4, 0.8),
            FuzzyTermTriangular(0.2, 0.6, 1),
        ])

    context.set_fuzzy_sets(fuzzy_sets)

    args = [
        "__data_name",
        1,
        2,
        1,
        "../dataset/yeast_multi/yeast-10-1tra.dat",
        "../dataset/yeast_multi/yeast-10-1tra.dat",
    ]


    MoFGBMLBasicMain.main(args)
    assert True
