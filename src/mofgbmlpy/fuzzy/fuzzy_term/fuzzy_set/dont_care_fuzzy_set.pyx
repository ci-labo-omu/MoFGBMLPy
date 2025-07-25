from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet

cdef class DontCareFuzzySet(FuzzySet):
    """Don't care fuzzy set """

    def __cinit__(self, int id, bint do_init=True):
        """Constructor

        Args:
            id (int): ID of the fuzzy set
            do_init (bool): If True, the object is initialized, otherwise it is not
        """

        if not do_init:
            self.ptr = NULL
            return

        self.ptr = new DontCareFuzzySetCpp(id)

