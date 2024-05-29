import pytest
from src.fuzzy.context.context import Context
from src.fuzzy.fuzzy_term.fuzzy_term_triangular import FuzzyTermTriangular


def test_get_instance():
    ref1 = Context.get_instance()
    ref2 = Context.get_instance()
    assert ref1 is ref2


def gen_fuzzy_sets_example():
    fuzzy_set_1 = [
        FuzzyTermTriangular(0, 0.4, 0.8),
        FuzzyTermTriangular(0.2, 0.6, 1),
    ]

    fuzzy_set_2 = [
        FuzzyTermTriangular(0, 0.4, 0.8),
        FuzzyTermTriangular(0.2, 0.6, 1),
    ]

    return [fuzzy_set_1, fuzzy_set_2]


def test_set_fuzzy_sets():
    try:
        fuzzy_sets = gen_fuzzy_sets_example()
        context = Context.get_instance()
        context.set_fuzzy_sets(fuzzy_sets)
        assert True
    except:
        assert False


def test_get_fuzzy_set():
    fuzzy_sets = gen_fuzzy_sets_example()
    context = Context.get_instance()
    context.set_fuzzy_sets(fuzzy_sets)

    assert context.get_fuzzy_set(0) is fuzzy_sets[0] and context.get_fuzzy_set(1) is fuzzy_sets[1]


def test_get_fuzzy_set_num():
    fuzzy_sets = gen_fuzzy_sets_example()
    context = Context.get_instance()
    context.set_fuzzy_sets(fuzzy_sets)

    assert context.get_fuzzy_get_num(0) == 2 and context.get_fuzzy_get_num(1) == 2


@pytest.mark.parametrize(('params', 'expected'), [((0, 0, 0), 0), ((0.4, 0, 0), 1), ((1, 0, 0), 0), ((0.2, 0, 0), 0.5),
                                                  ((0, 0, 1), 0), ((0.6, 0, 1), 1), ((1, 0, 1), 0), ((0.8, 0, 1), 0.5)])
def test_get_membership_value(params, expected):
    fuzzy_sets = gen_fuzzy_sets_example()
    context = Context.get_instance()
    context.clear()
    context.set_fuzzy_sets(fuzzy_sets)

    assert pytest.approx(context.get_membership_value(params[0], params[1], params[2])) == expected


def test_get_num_dim():
    fuzzy_sets = gen_fuzzy_sets_example()
    context = Context.get_instance()
    context.clear()
    context.set_fuzzy_sets(fuzzy_sets)

    assert context.get_num_dim() == 2


def test_clear():
    fuzzy_sets = gen_fuzzy_sets_example()
    context = Context.get_instance()
    context.clear()
    context.set_fuzzy_sets(fuzzy_sets)
    s1 = context.get_num_dim()
    context.clear()
    s2 = context.get_num_dim()

    assert s1 != 0 and s2 == 0
