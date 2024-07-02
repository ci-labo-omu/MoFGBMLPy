import math
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
cimport numpy as cnp

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set cimport FuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.linguistic_variable cimport LinguisticVariable
import simpful

from mofgbmlpy.fuzzy.fuzzy_term.simpful_linguistic_variable_adaptor import SimpfulLinguisticVariableAdaptor

cdef class Knowledge:
    def __init__(self, fuzzy_sets=None):
        if fuzzy_sets is None:
            self.__fuzzy_sets = np.empty(0, dtype=object)
        else:
            self.__fuzzy_sets = fuzzy_sets

    cpdef LinguisticVariable get_fuzzy_variable(self, int dim):
        return self.__fuzzy_sets[dim]

    cpdef FuzzySet get_fuzzy_set(self, int dim, int fuzzy_set_id):
        cdef LinguisticVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.size == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        cdef LinguisticVariable var = fuzzy_sets[dim]
        return var.get_fuzzy_set(fuzzy_set_id)

    cpdef int get_num_fuzzy_sets(self, int dim):
        cdef LinguisticVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.size == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        cdef LinguisticVariable var = fuzzy_sets[dim]
        return var.get_length()

    cpdef void set_fuzzy_sets(self, LinguisticVariable[:] fuzzy_sets):
        # cdef LinguisticVariable[:] self_fuzzy_sets = self.__fuzzy_sets
        #     if self_fuzzy_sets is None or self_fuzzy_sets.size == 0:
        if self.__fuzzy_sets is not None and len(self.__fuzzy_sets) != 0:
            # with cython.gil:
            raise Exception("You can't overwrite fuzzy sets. You must call clear before doing so")

        self.__fuzzy_sets = fuzzy_sets

    cpdef LinguisticVariable[:] get_fuzzy_sets(self):
        cdef LinguisticVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.size == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        return fuzzy_sets

    cpdef double get_membership_value_py(self, double attribute_value, int dim, int fuzzy_set_id):
        cdef LinguisticVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.size == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")
        cdef LinguisticVariable var = fuzzy_sets[dim]
        return var.get_membership_value(fuzzy_set_id, attribute_value)

    cdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_id):
        cdef LinguisticVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.size == 0:
            return -1
        cdef LinguisticVariable var = fuzzy_sets[dim]
        return var.get_membership_value(fuzzy_set_id, attribute_value)

    cpdef int get_num_dim(self):
        cdef LinguisticVariable[:] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None:
            return 0
        return fuzzy_sets.size

    cpdef void clear(self):
        self.__fuzzy_sets = np.empty(0, dtype=object)

    def plot_one_fuzzy_set(self, dim_i, fuzzy_set_id):
        temp = SimpfulLinguisticVariableAdaptor.create_var_single_fuzzy_set(self.__fuzzy_sets[dim_i], fuzzy_set_id)
        temp.plot()

    def draw_one_fuzzy_set(self, dim_i, fuzzy_set_id, ax):
        temp = SimpfulLinguisticVariableAdaptor.create_var_single_fuzzy_set(self.__fuzzy_sets[dim_i], fuzzy_set_id)
        ax.set_title(self.get_fuzzy_set(dim_i, fuzzy_set_id).get_term())
        return temp.draw(ax)

    def plot_one_dim_unique_graph(self, dim_i):
        temp = SimpfulLinguisticVariableAdaptor(self.__fuzzy_sets[dim_i])
        temp.plot()

    def draw_one_dim(self, dim_i, ax):
        temp = SimpfulLinguisticVariableAdaptor(self.__fuzzy_sets[dim_i])
        return temp.draw(ax)

    def plot_one_dim_separate_graphs(self, dim_i):
        plots_per_line = math.ceil(math.sqrt(self.get_num_fuzzy_sets(dim_i)))

        fig, axs = plt.subplots(plots_per_line, plots_per_line)
        fig.tight_layout(pad=7.0)
        fig.set_size_inches(5 * plots_per_line, 5 * plots_per_line)

        x, y = 0, 0
        for i in range(self.get_num_fuzzy_sets(dim_i)):
            axs[x, y] = self.draw_one_fuzzy_set(dim_i, i, axs[x, y])
            axs[x, y].legend(loc='right', bbox_to_anchor=(1.8, 0.5))
            y += 1
            if y >= plots_per_line:
                y = 0
                x += 1
        fig.suptitle("Variable "+self.get_fuzzy_variable(dim_i).get_concept())
        fig.show()

    def plot_all_dim(self):
        plots_per_line = math.ceil(math.sqrt(self.get_num_dim()))

        fig, axs = plt.subplots(plots_per_line, plots_per_line)
        fig.tight_layout(pad=7.0)
        fig.set_size_inches(5 * plots_per_line, 5 * plots_per_line)

        x, y = 0, 0
        for i in range(self.get_num_dim()):
            axs[x, y] = self.draw_one_dim(i, axs[x, y])
            axs[x, y].legend(loc='right', bbox_to_anchor=(1.8, 0.5))
            y += 1
            if y >= plots_per_line:
                y = 0
                x += 1

        fig.show()

    cpdef double get_support(self, int  dim, int  fuzzy_set_id):
        return self.get_fuzzy_variable(dim).get_support(fuzzy_set_id)

    def __repr__(self):
        txt = ""
        for i in range(self.get_num_dim()):
            txt = f"{txt}{str(self.__fuzzy_sets[i])}\n"
        return txt

    def __deepcopy__(self, memo={}):
        cdef LinguisticVariable[:] fuzzy_sets = self.__fuzzy_sets
        cdef LinguisticVariable[:] fuzzy_sets_copy = np.empty(fuzzy_sets.size, dtype=object)
        cdef int i

        for i in range(fuzzy_sets.size):
            fuzzy_sets_copy[i] = deepcopy(fuzzy_sets[i])

        new_knowledge = Knowledge(fuzzy_sets_copy)
        memo[id(self)] = new_knowledge

        return new_knowledge
