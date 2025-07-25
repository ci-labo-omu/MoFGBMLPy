from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet


cdef class RectangularFuzzySet(FuzzySet):
    """Rectangular fuzzy set """
    def __cinit__(self, float left, float right, int id, str term, bint do_init=True):
        """Constructor

        Args:
            left (float): X coordinate of the leftmost side of the rectangle: membership is equals to 0 before this point and 1 after it
            right (float): X coordinate of the leftmost side of the rectangle: membership is equals to 0 after this point and 1 before it
            id (int): ID of the fuzzy set
            term (str): Name of the fuzzy set (e.g. small)
            do_init (bool): If True, the object is initialized, otherwise it is not
        """

        if not do_init:
            self.ptr = NULL
            return

        if term is None:
            raise TypeError("Term cannot be None")

        self.ptr = new RectangularFuzzySetCpp(left, right, id, term.encode("utf-8"))