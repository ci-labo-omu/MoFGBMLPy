import copy

import numpy as np


def exists_in_sols(sol, sols_list):
    """Check if a solution exists in a list of solutions.

    Args:
        sol (MichiganSolution): The solution to check
        sols_list (list): List of solutions to check against

    Returns:
        bool: True if the solution exists in the list, False otherwise
    """
    for filtered_sol in sols_list:
        if sol[0] == filtered_sol[0]:
            return True
    return False


def remove_duplicates(sols):
    """Remove duplicate solutions from a list of solutions.

    Args:
        sols (list): List of solutions to filter

    Returns:
        np.array: Array of unique solutions
    """
    filtered_sols = []
    for sol in sols:
        if not exists_in_sols(sol, filtered_sols):
            filtered_sols.append(sol)
    return np.array(filtered_sols, dtype=object)


def append_rule_classifier(initial_classifier, new_rule, train_set=None, deepcopy=True):
    """Append a rule to a classifier and return the new classifier.

    Args:
        initial_classifier (PittsburghSolution): The initial classifier to which the rule will be appended.
        new_rule (MichiganSolution): The rule to append to the classifier.
        train_set (Dataset): The training dataset to update the classifier with if needed.
        deepcopy (bool): Whether to create a deep copy of the initial classifier before modifying it. Default is True.

    Returns:
        PittsburghSolution: A new classifier with the new rule appended
    """
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
