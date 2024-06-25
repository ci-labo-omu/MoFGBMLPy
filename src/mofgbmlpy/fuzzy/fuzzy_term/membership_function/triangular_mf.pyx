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

    cpdef double get_value(self, double x):
        if x < self.__center:
            if self.__left != self.__center:
                return (x - self.__left) * (1 / (self.__center - self.__left))
            else:
                return 1
        else:
            if self.__center != self.__right:
                return 1 + (x - self.__center) * (-1 / (self.__right - self.__center))
            else:
                return 1

    def __str__(self):
        return "<Triangular MF (%f, %f, %f)>" % (self.__left, self.__center, self.__right)