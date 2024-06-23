cimport numpy as cnp


ctypedef fused double_or_double_array:
    double
    cnp.ndarray[double, ndim=1]

ctypedef fused int_or_int_array:
    int
    cnp.ndarray[int, ndim=1]