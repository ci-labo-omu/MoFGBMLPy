import functools

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3


def dash_case_to_class_name(txt):
    """Convert a text in dash case format to a class name format. e.g. an-example becomes AnExample

    Args:
        txt (str): text to be converted

    Returns:
        str: New text
    """
    parts = txt.split('-')
    parts = [p.capitalize() for p in parts]
    return ''.join(parts)


def dash_case_to_snake_case(txt):
    """Convert a text in dash case format to snake case. e.g. an-example becomes an_example

    Args:
        txt (str): text to be converted

    Returns:
        str: New text
    """
    return txt.replace('-', '_')


def get_algo(algo_name, **kwargs):
    algo_args = {
        "eliminate_duplicates": False,
        "save_history": True,
    }

    for arg in ["pop_size", "sampling", "crossover", "repair", "mutation"]:
        if arg not in kwargs:
            raise Exception(f"Missing argument {arg}")
        algo_args[arg] = kwargs[arg]

    algos = {
        "nsga2": {"class": NSGA2, "additional_args": ["n_offsprings"]},
        "nsga3": NSGA3,
        "moead": MOEAD
    }

    if algo_name not in algos:
        raise ValueError("Unknown algo name")

    for arg in algos[algo_name]["additional_args"]:
        if arg not in kwargs:
            raise Exception(f"Missing argument {arg}")
        algo_args[arg] = kwargs[arg]

    return algos[algo_name]["class"](algo_args)
