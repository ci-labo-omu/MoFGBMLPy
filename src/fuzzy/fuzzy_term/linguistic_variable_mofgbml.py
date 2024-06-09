from simpful.simpful import LinguisticVariable


class LinguisticVariableMoFGBML(LinguisticVariable):
    def __init__(self, fuzzy_sets=None, name=None):
        if fuzzy_sets is None:
            fuzzy_sets = []
        super().__init__(fuzzy_sets, concept=name)

    def add_fuzzy_set(self, fuzzy_set):
        self._FSlist.append(fuzzy_set)

    def get_membership_value(self, fuzzy_set_index, x):
        if fuzzy_set_index > len(self._FSlist):
           raise Exception(f"{fuzzy_set_index} is out of range (>= {len(self._FSlist)})")
        return self._FSlist[fuzzy_set_index].get_value(x)

    def get_length(self):
        return len(self._FSlist)
