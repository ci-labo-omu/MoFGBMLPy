import xml.etree.cElementTree as xml_tree

import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF

cdef class DontCareMF(AbstractMF):
    cdef double get_value(self, double _):
        return 1.0

    def __str__(self):
        return "<Dont Care MF>"

    def to_xml(self):
        root = xml_tree.Element("membershipFunction")
        root.text = "Dont care"
        return root

    cpdef double[:,:] get_plot_points(self):
        return np.array([[0,1],[1,1]], dtype=np.float64)