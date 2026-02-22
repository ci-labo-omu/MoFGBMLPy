import xml.etree.cElementTree as xml_tree
import copy

import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet


cdef class FuzzyVariable:
    """Fuzzy variable

    Attributes:
        __fuzzy_sets (FuzzySet[]): List of the fuzzy sets of this variable
        __name (str): Name of the fuzzy variable (e.g. Petal length)
        __domain (float[]): Domain of the values in the variable (uses only for plotting purposes for now)
    """
    def __init__(self, FuzzySet[:] fuzzy_sets, str name="unnamed_var", domain=None):
        """Constructor

        Args:
            fuzzy_sets (FuzzySet[]): List of the fuzzy sets of this variable
            name (str): Name of the fuzzy variable (e.g. Petal length)
            domain (float[]): List of the fuzzy sets of this variable

        Raises:
            Exception: None name or empty or None fuzzy sets array
        """
        if name is None:
            raise TypeError("name can't be done")
        elif fuzzy_sets is None:
            raise TypeError("fuzzy_sets must not be None")
        elif len(fuzzy_sets) == 0:
            raise ValueError("fuzzy_sets must have at least one element")

        self.__fuzzy_sets = fuzzy_sets
        self.__name = name

        if domain is None:
            self.__domain = np.array([0.0, 1.0], dtype=np.float32)
        else:
            if len(domain) != 2:
                raise ValueError("domain must be an array of float size 2 (min, max)")
            elif domain[0] > domain[1]:
                raise ValueError("domain's first value must be lesser than the second one")
            self.__domain = domain

    cpdef str get_name(self):
        """Get the name of the variable
        
        Returns:
            Variable's name
        """
        return self.__name

    cdef float get_membership_value(self, int fuzzy_set_index, float x):
        """Get the membership value for the value x with the given fuzzy set (Accessible only from Cython code)
        
        Args:
            fuzzy_set_index (int): Index of the fuzzy set used to compute the membership value 
            x (float): Value whose membership value is computed

        Returns:
            float: Membership value
        
        Raises:
            Exception: The index is out of range
        """
        if fuzzy_set_index >= self.__fuzzy_sets.shape[0]:
            raise IndexError(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")
        cdef FuzzySet fuzzy_set = self.__fuzzy_sets[fuzzy_set_index]
        return fuzzy_set.get_membership_value(x)

    def get_membership_value_py(self, int fuzzy_set_index, float x):
        """Get the membership value for the value x with the given fuzzy set

        Args:
            fuzzy_set_index (int): Index of the fuzzy set used to compute the membership value
            x (float): Value whose membership value is computed

        Returns:
            float: Membership value
        """
        return self.get_membership_value(fuzzy_set_index, x)

    cpdef int get_length(self):
        """Get the length of the fuzzy sets array (number of fuzzy sets for this variable including don't care)
        
        Returns:
            int: Number of furry sets
        """
        return len(self.__fuzzy_sets)

    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index):
        """Get the fuzzy set at the given index
        
        Args:
            fuzzy_set_index (int): Index where the fuzzy set is fetched 

        Returns:
            FuzzySet: Fuzzy set fetched
       
        Raises:
            Exception: The index is out of range
        """
        if fuzzy_set_index >= self.__fuzzy_sets.shape[0]:
            raise IndexError(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")
        return self.__fuzzy_sets[fuzzy_set_index]

    cpdef float get_support(self, int fuzzy_set_index):
        """Get the support value of a fuzzy set. This value corresponds to the area covered by the membership function in the search space (e.g. for don't care in [0,1] it's 1)
        
        Args:
            fuzzy_set_index (int): Index of the fuzzy set whose support value is computed 

        Returns:
            float: Support value
        Raises:
            Exception: The index is out of range
        """
        if fuzzy_set_index >= self.__fuzzy_sets.shape[0]:
            raise IndexError(f"{fuzzy_set_index} is out of range (>= {len(self.__fuzzy_sets)})")

        cdef FuzzySet fuzzy_set = self.__fuzzy_sets[fuzzy_set_index]
        return fuzzy_set.get_support(self.__domain[0], self.__domain[1])

    cpdef get_fuzzy_sets(self):
        """Get the array of fuzzy sets of this variable
        
        Returns:
            FuzzySets[]: Array of fuzzy sets
        """
        return self.__fuzzy_sets

    cpdef get_support_values(self):
        """Get all the support values (one per fuzzy set) in an array
        
        Returns:
            float[]: Array of support values
        """
        cdef float[:] support_values = np.empty(len(self.__fuzzy_sets))
        cdef int i
        cdef FuzzySet fuzzy_set

        for i in range(len(self.__fuzzy_sets)):
            fuzzy_set = self.__fuzzy_sets[i]
            support_values[i] = fuzzy_set.get_support(self.__domain[0], self.__domain[1])
        return support_values

    cpdef get_domain(self):
        """Get the domain of this variable (e.g [0,1])
        
        Returns:
            float[]: Domain of this variable (min and max values)
        """
        return self.__domain

    def get_plot(self, ax, line_width=2):
        """Draw the fuzzy variable fuzzy sets on the given matplotlib Axes object

        Args:
            ax (matplotlib.axes.Axes): Axes object
            line_width (int): Line width used to draw the fuzzy sets

        Returns:
            matplotlib.axes.Axes: The axes object where we drew
        """
        cdef int i
        cdef FuzzySet fuzzy_set
        cdef cnp.ndarray[float, ndim=2] points

        ax.set_title(self.get_name())
        for i in range(self.get_length()):
            fuzzy_set = self.get_fuzzy_set(i)
            points = fuzzy_set.get_function().get_plot_points(self.__domain[0], self.__domain[1])
            ax.plot(points[:,0], points[:,1], label=fuzzy_set.get_term(), linewidth=line_width)

        ax.legend(loc="upper right")
        ax.set_xlim(self.get_domain())
        ax.set_ylim([0, 1.1])

        return ax

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        txt = f"Fuzzy variable for {self.__name}:\n"
        cdef int i
        for i in range(len(self.__fuzzy_sets)):
            txt += f"\t{self.__fuzzy_sets[i]}\n"
        return txt

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef FuzzySet[:] fuzzy_sets_copy = np.empty(self.__fuzzy_sets.shape[0], dtype=object)
        cdef int i

        for i in range(fuzzy_sets_copy.shape[0]):
            fuzzy_sets_copy[i] = copy.deepcopy(self.__fuzzy_sets[i])

        cdef FuzzyVariable new_object = FuzzyVariable(fuzzy_sets_copy, self.__name, np.copy(self.__domain))
        memo[id(self)] = new_object
        return new_object

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("fuzzySets")

        for i in range(self.get_length()):
            root.append(self.get_fuzzy_set(i).to_xml())

        return root

    @staticmethod
    def from_xml(xml_element):
        """Create a FuzzyVariable object from its XML representation

        Args:
            xml_element (xml.etree.ElementTree): XML element representing the fuzzy variable

        Returns:
            FuzzyVariable: FuzzyVariable object created from the XML element
        """
        imported_fuzzy_sets = xml_element.findall("fuzzyTerm")
        cdef FuzzySet[:] fuzzy_sets = np.empty(len(imported_fuzzy_sets), dtype=object)

        for i in range(len(imported_fuzzy_sets)):
            fuzzy_sets[i] = FuzzySet.from_xml(imported_fuzzy_sets[i])

        return FuzzyVariable(fuzzy_sets)

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, FuzzyVariable):
            return False


        return (np.array_equal(self.__fuzzy_sets, other.get_fuzzy_sets()) and
                np.array_equal(self.__domain, other.get_domain()) and
                self.__name == other.get_name())