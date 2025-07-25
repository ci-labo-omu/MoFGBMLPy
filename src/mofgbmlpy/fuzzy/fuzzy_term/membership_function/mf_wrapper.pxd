from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF, AbstractMFCpp

cdef AbstractMF wrap_mf(AbstractMFCpp * wrapped_ptr)
