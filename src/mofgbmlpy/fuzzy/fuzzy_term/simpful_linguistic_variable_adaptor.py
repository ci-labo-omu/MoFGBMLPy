import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.linguistic_variable import LinguisticVariable
import simpful

from mofgbmlpy.fuzzy.fuzzy_term.simpful_fuzzy_set_adaptor import SimpfulFuzzySetAdaptor


class SimpfulLinguisticVariableAdaptor(simpful.LinguisticVariable):
    def __init__(self, var):
        fuzzy_sets = []
        for set in var.get_fuzzy_sets():
            fuzzy_sets.append(SimpfulFuzzySetAdaptor(set))

        super().__init__(fuzzy_sets, var.get_concept(), var.get_domain())

    @staticmethod
    def create_var_single_fuzzy_set(var, fuzzy_set_id):
        temp = LinguisticVariable(np.array([var.get_fuzzy_set(fuzzy_set_id)], dtype=object),
                                  var.get_concept(),
                                  var.get_domain())
        return SimpfulLinguisticVariableAdaptor(temp)
