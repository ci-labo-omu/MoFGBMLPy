cdef class AbstractMF:
    cpdef double get_value(self, double x):
        raise Exception("This class is abstract")

    def __str__(self):
        return "Abstract membership function"
