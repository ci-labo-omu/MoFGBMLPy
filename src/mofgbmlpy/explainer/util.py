import numpy as np
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from pymoo.core.population import Population

from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain


def exists_in_sols(sol, sols_list):
    for filtered_sol in sols_list:
        if sol[0] == filtered_sol[0]:
            return True
    return False


def remove_duplicates(sols):
    filtered_sols = []
    for sol in sols:
        if not exists_in_sols(sol, filtered_sols):
            filtered_sols.append(sol)
    return np.array(filtered_sols, dtype=object)

