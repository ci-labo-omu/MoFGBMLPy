import cython

cdef class AbstractMF:
    cdef double get_value(self, double x)
    cpdef double[:,:] get_plot_points(self)