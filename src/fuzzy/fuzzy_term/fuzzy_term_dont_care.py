from src.fuzzy.fuzzy_term.abstract_fuzzy_term import AbstractFuzzyTerm


class FuzzyTermDontCare(AbstractFuzzyTerm):
    def __init__(self):
        super().__init__()

    def get_membership_value(self, x):
        return 1.0
