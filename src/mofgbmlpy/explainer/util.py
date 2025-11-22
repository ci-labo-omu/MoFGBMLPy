import copy

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


def append_rule_classifier(initial_classifier, new_rule, train_set=None, deepcopy=True):
    if deepcopy:
        new_cl = copy.deepcopy(initial_classifier)
    else:
        new_cl = initial_classifier

    old_vars = new_cl.get_vars()
    new_vars = np.empty(len(old_vars) + 1, dtype=object)
    for i in range(len(old_vars)):
        new_vars[i] = old_vars[i]
    new_vars[-1] = new_rule
    new_cl.set_vars(new_vars)
    new_cl.learning()
    if train_set is not None:
        new_cl.update_winners_and_errors(train_set)
    return new_cl