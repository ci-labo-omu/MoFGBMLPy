from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
import cython

cdef class TriangularMF(AbstractMF):
    cdef double __left
    cdef double __center
    cdef double __right

    cdef double get_value(self, double x)
    cpdef double[:,:] get_plot_points(self)