import xml.etree.cElementTree as xml_tree

import numpy as np

cdef class AbstractMF:
    def __init__(self, double[:] params, bint are_params_points_flag=True):
        self._params = params
        if self._params is None:
            self._params = np.empty(0)
        self._are_params_points_flag = are_params_points_flag

    cdef double get_value(self, double x):
        # with cython.gil:
        raise Exception("This class is abstract")

    def get_value_py(self, double x):
        return self.get_value(x)

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

    cpdef cnp.ndarray[double, ndim=1] get_param_range(self, int index, double x_min=0, double x_max=1):
        raise Exception("This class is abstract")

    cpdef bint is_param_value_valid(self, int index, double value, double x_min=0, double x_max=1):
        cdef double[:] val_range = self.get_param_range(index, x_min, x_max)
        return val_range[0] <= value and value <= val_range[1]

    cpdef bint are_params_points(self):
        return self._are_params_points_flag

    cpdef void set_param_value(self, int index, double value, double x_min=0, double x_max=1):
        if self.is_param_value_valid(index, value, x_min, x_max):
            self._params[index] = value
        else:
            raise Exception("Invalid index or value")

    def __deepcopy__(self, memo={}):
        raise Exception("This class is abstract")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return np.array_equal(self._params, other.get_params())

    cpdef cnp.ndarray[double, ndim=2] get_plot_points(self, double x_min=0, double x_max=1):
        raise Exception("This class is abstract")