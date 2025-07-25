from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF, AbstractMFCpp
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.rectangular_mf cimport RectangularMFCpp, RectangularMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.triangular_mf cimport TriangularMFCpp, TriangularMF
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.dont_care_mf cimport DontCareMFCpp, DontCareMF

from libcpp.cast cimport dynamic_cast

ctypedef DontCareMFCpp* DCPtr
ctypedef TriangularMFCpp* TrPtr
ctypedef RectangularMFCpp* RectPtr

cdef AbstractMF wrap_mf(AbstractMFCpp * wrapped_ptr):
    cdef DCPtr dc_ptr = dynamic_cast[DCPtr](wrapped_ptr)
    if dc_ptr != NULL:
        return DontCareMF.wrap(dc_ptr)

    cdef TrPtr tr_ptr = dynamic_cast[TrPtr](wrapped_ptr)
    if tr_ptr != NULL:
        return TriangularMF.wrap(tr_ptr)

    cdef RectPtr rect_ptr = dynamic_cast[RectPtr](wrapped_ptr)
    if rect_ptr != NULL:
        return RectangularMF.wrap(rect_ptr)

    raise TypeError("Unknown MF type")
