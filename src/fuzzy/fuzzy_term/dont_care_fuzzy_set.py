from simpful.fuzzy_sets import FuzzySet
from simpful.fuzzy_sets import MF_object


class DontCareFuzzySet(FuzzySet):
    class DontCareMF(MF_object):
        def _execute(self, _):
            return 1.0

        def __repr__(self):
            return "<Dont Care MF>"

    def __init__(self):
        dont_care_mf = DontCareFuzzySet.DontCareMF()
        super().__init__(function=dont_care_mf, term="DC")
