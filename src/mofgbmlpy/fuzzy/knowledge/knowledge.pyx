import math

from matplotlib import pyplot as plt
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.linguistic_variable import LinguisticVariable

cdef class Knowledge:
    def __init__(self):
        self.__fuzzy_sets = []

    cpdef object get_fuzzy_variable(self, int dim):
        return self.__fuzzy_sets[dim]

    cpdef object get_fuzzy_set(self, int dim, int fuzzy_set_id):
        if self.__fuzzy_sets is None or len(self.__fuzzy_sets) == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        return self.__fuzzy_sets[dim].get_fuzzy_set(fuzzy_set_id)

    cpdef int get_num_fuzzy_sets(self, int dim):
        if self.__fuzzy_sets is None or len(self.__fuzzy_sets) == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        return self.__fuzzy_sets[dim].get_length()

    cpdef void set_fuzzy_sets(self, cnp.ndarray[object, ndim=1] fuzzy_sets):
        if self.__fuzzy_sets is not None and len(self.__fuzzy_sets) != 0:
            # with cython.gil:
            raise Exception("You can't overwrite fuzzy sets. You must call clear before doing so")

        self.__fuzzy_sets = fuzzy_sets

    cpdef cnp.ndarray[object, ndim=1] get_fuzzy_sets(self):
        if self.__fuzzy_sets is None or len(self.__fuzzy_sets) == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        return self.__fuzzy_sets

    cpdef double get_membership_value_py(self, double attribute_value, int dim, int fuzzy_set_id):
        if self.__fuzzy_sets is None or self.__fuzzy_sets.size == 0:
            # with cython.gil:
            raise Exception("Context is not yet initialized (no fuzzy set)")
        return self.__fuzzy_sets[dim].get_membership_value(fuzzy_set_id, attribute_value)

    cdef double get_membership_value(self, double attribute_value, int dim, int fuzzy_set_id):
        cdef cnp.ndarray[object, ndim=1] fuzzy_sets = self.__fuzzy_sets
        if fuzzy_sets is None or fuzzy_sets.size == 0:
            return -1
        return fuzzy_sets[dim].get_membership_value(fuzzy_set_id, attribute_value)

    cpdef int  get_num_dim(self):
        if self.__fuzzy_sets is None:
            return 0
        return len(self.__fuzzy_sets)

    cpdef void clear(self):
        self.__fuzzy_sets = []

    # def plot_one_fuzzy_set(self, dim_i, fuzzy_set_id):
    #     linguistic_var = self.get_fuzzy_variable(dim_i)
    #     temp = LinguisticVariable([linguistic_var.get_fuzzy_set(fuzzy_set_id)],
    #                               linguistic_var.get_concept(),
    #                               linguistic_var.get_universe_of_discourse())
    #     temp.plot()
    #
    # def draw_one_fuzzy_set(self, dim_i, fuzzy_set_id, ax):
    #     linguistic_var = self.get_fuzzy_variable(dim_i)
    #     temp = LinguisticVariable([linguistic_var.get_fuzzy_set(fuzzy_set_id)],
    #                               linguistic_var.get_concept(),
    #                               linguistic_var.get_universe_of_discourse())
    #     ax.set_title(self.get_fuzzy_set(dim_i, fuzzy_set_id).get_term())
    #     return temp.draw(ax)
    #
    # def plot_one_dim_unique_graph(self, dim_i):
    #     self.__fuzzy_sets[dim_i].plot()
    #
    # def plot_one_dim_separate_graphs(self, dim_i):
    #     plots_per_line = math.ceil(math.sqrt(self.get_num_fuzzy_sets(dim_i)))
    #
    #     fig, axs = plt.subplots(plots_per_line, plots_per_line)
    #     fig.tight_layout(pad=7.0)
    #     fig.set_size_inches(5 * plots_per_line, 5 * plots_per_line)
    #
    #     x, y = 0, 0
    #     for i in range(self.get_num_fuzzy_sets(dim_i)):
    #         axs[x, y] = self.draw_one_fuzzy_set(dim_i, i, axs[x, y])
    #         axs[x, y].legend(loc='right', bbox_to_anchor=(1.8, 0.5))
    #         y += 1
    #         if y >= plots_per_line:
    #             y = 0
    #             x += 1
    #     fig.suptitle("Variable "+self.get_fuzzy_variable(dim_i).get_concept())
    #     fig.show()
    #
    # def plot_all_dim(self):
    #     plots_per_line = math.ceil(math.sqrt(self.get_num_dim()))
    #
    #     fig, axs = plt.subplots(plots_per_line, plots_per_line)
    #     fig.tight_layout(pad=7.0)
    #     fig.set_size_inches(5 * plots_per_line, 5 * plots_per_line)
    #
    #     x, y = 0, 0
    #     for i in range(self.get_num_dim()):
    #         axs[x, y] = self.__fuzzy_sets[i].draw(axs[x, y])
    #         axs[x, y].legend(loc='right', bbox_to_anchor=(1.8, 0.5))
    #         y += 1
    #         if y >= plots_per_line:
    #             y = 0
    #             x += 1
    #
    #     fig.show()

    cpdef double get_support(self, int  dim, int  fuzzy_set_id):
        return self.get_fuzzy_variable(dim).get_support(fuzzy_set_id)

    def __str__(self):
        txt = ""
        for i in range(self.get_num_dim()):
            txt = f"{txt}{str(self.__fuzzy_sets[i])}\n"
        return txt
