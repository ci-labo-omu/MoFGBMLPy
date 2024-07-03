cdef class AbstractMF:
    cdef double get_value(self, double x):
        # with cython.gil:
        raise Exception("This class is abstract")

    def __str__(self):
        return "Abstract membership function"

    def to_xml(self):
        raise Exception("This class is abstract")
