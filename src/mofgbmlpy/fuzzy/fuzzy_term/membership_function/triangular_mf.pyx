from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF


cdef class TriangularMF(AbstractMF):
    def __init__(self, left=0, center=0.5, right=1):
        self.__left = left
        self.__center = center
        self.__right = right

        if left > center:
            # with cython.gil:
            raise Exception(f"Error in triangular membership function: left={left:.2f} should be <= center={center:.2f}")
        elif center > right:
            # with cython.gil:
            raise Exception(f"Error in triangular membership function: center={center:.2f} should be <= right={right:.2f}")

    cdef double get_value(self, double x):
        if x < self.__left or x > self.__right:
            return 0
        elif x == self.__center:
            return 1
        elif x < self.__center:
            return (x - self.__left) / (self.__center - self.__left)
        else:
            return 1 + (x - self.__center) / (self.__right - self.__center)

    def __str__(self):
        return "<Triangular MF (%f, %f, %f)>" % (self.__left, self.__center, self.__right)