import xml.etree.cElementTree as xml_tree

import numpy as np

cdef class AbstractMF:
    def __init__(self, params, are_params_points=True):
        self._params = params
        self._are_params_points = are_params_points

    cdef double get_value(self, double x):
        # with cython.gil:
        raise Exception("This class is abstract")

    def __str__(self):
        return "Abstract membership function"

    def to_xml(self):
        root = xml_tree.Element("membershipFunction")

        if len(self._params) != 0:
            params_set = xml_tree.SubElement(root, "parameterSet")

            for i in range(len(self._params)):
                param = xml_tree.SubElement(params_set, "parameterSet")
                param.text = str(self._params[i])
                param.set("id", str(i))

        return root

    cpdef cnp.ndarray[double, ndim=1] get_params(self):
        return np.array(self._params)

    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index):
        raise Exception("This class is abstract")

    cpdef bint is_param_value_valid(self, int index, double value):
        raise Exception("This class is abstract")

    cpdef void set_param_value(self, int index, double value):
        if self.is_param_value_valid(index, value):
            self._params[index] = value
        else:
            raise Exception("Invalid index or value")

    def __deepcopy__(self, memo={}):
        raise Exception("This class is abstract")

    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self):
        raise Exception("This class is abstract")