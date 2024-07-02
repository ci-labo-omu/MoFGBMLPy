import simpful


class SimpfulFuzzySetAdaptor(simpful.FuzzySet):
    def __init__(self, fuzzy_set):
        super().__init__(function=fuzzy_set.get_function_callable(), term=fuzzy_set.get_term())
