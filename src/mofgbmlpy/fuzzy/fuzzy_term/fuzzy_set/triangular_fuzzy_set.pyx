from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet


cdef class TriangularFuzzySet(FuzzySet):
    def __cinit__(self, float left, float center, float right, int id, str term, do_init=True):
        """Constructor

        Args:
            left (float): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 before it
            center (float): X coordinate of the vertex in the center of the triangle: membership is equals to 1 at this point
            right (float): X coordinate of the leftmost vertex of the triangle: membership is equals to 0 after it
            id (int): ID of the fuzzy set
            term (str): Name of the fuzzy set (e.g. small)
            do_init (bool): If True, the object is initialized, otherwise it is not
        """

        if not do_init:
            return

        if term is None:
            raise TypeError("Term cannot be None")

        self.ptr = new TriangularFuzzySetCpp(
            left, center, right, id, term.encode("utf-8")
        )