import xml.etree.cElementTree as xml_tree
import copy

import numpy as np
cimport numpy as cnp
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.fuzzy_set cimport FuzzySet, FuzzySetCpp


cdef class FuzzyVariable:
    """Fuzzy variable

    Attributes:
        __fuzzy_sets (FuzzySet[]): List of the fuzzy sets of this variable
        __name (str): Name of the fuzzy variable (e.g. Petal length)
        __domain (float[]): Domain of the values in the variable (uses only for plotting purposes for now)
    """
    def __cinit__(self, FuzzySet[:] fuzzy_sets, str name="unnamed_var", float[:] domain=None, do_init=True):
        """Constructor

        Args:
            fuzzy_sets (FuzzySet[]): List of the fuzzy sets of this variable
            name (str): Name of the fuzzy variable (e.g. Petal length)
            domain (float[]): List of the fuzzy sets of this variable
            do_init (bool): If True, the object is initialized, otherwise it is not

        Raises:
            Exception: None name or empty or None fuzzy sets array
        """
        if not do_init:
            self.ptr = NULL
            return

        if name is None:
            raise TypeError("Name cannot be None")

        cdef vector[FuzzySetCpp*] cpp_fuzzy_sets
        if fuzzy_sets is not None and fuzzy_sets.shape[0] != 0:
            cpp_fuzzy_sets.reserve(fuzzy_sets.shape[0])
            for i in range(fuzzy_sets.shape[0]):
                cpp_fuzzy_sets.push_back(fuzzy_sets[i].ptr.clone())

        cdef vector[float] cpp_domain
        if domain is not None and domain.shape[0] != 0:
            cpp_domain.reserve(domain.shape[0])
            for i in range(domain.shape[0]):
                cpp_domain.push_back(domain[i])
        else:
            cpp_domain = {0.0, 1.0}

        self.ptr = new FuzzyVariableCpp(cpp_fuzzy_sets, name.encode("utf-8"), cpp_domain)

    def __dealloc__(self):
        """Destructor"""
        if self.ptr != NULL:
            del self.ptr

    cpdef str get_name(self):
        """Get the name of the variable
        
        Returns:
            Variable's name
        """
        return self.ptr.get_name().decode("utf-8")

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
        return self.ptr.get_membership_value(fuzzy_set_index, x)

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
        return self.ptr.get_length()

    cpdef FuzzySet get_fuzzy_set(self, int fuzzy_set_index):
        """Get the fuzzy set at the given index
        
        Args:
            fuzzy_set_index (int): Index where the fuzzy set is fetched 

        Returns:
            FuzzySet: Fuzzy set fetched
       
        Raises:
            Exception: The index is out of range
        """
        return FuzzySet.wrap(self.ptr.get_fuzzy_set(fuzzy_set_index))

    cpdef float get_support(self, int fuzzy_set_index):
        """Get the support value of a fuzzy set. This value corresponds to the area covered by the membership function in the search space (e.g. for don't care in [0,1] it's 1)
        
        Args:
            fuzzy_set_index (int): Index of the fuzzy set whose support value is computed 

        Returns:
            float: Support value
        Raises:
            Exception: The index is out of range
        """
        return self.ptr.get_support(fuzzy_set_index)

    cpdef get_fuzzy_sets(self):
        """Get the array of fuzzy sets of this variable
        
        Returns:
            FuzzySets[]: Array of fuzzy sets
        """
        cdef vector[FuzzySetCpp*] cpp_fuzzy_sets = self.ptr.get_fuzzy_sets()
        cdef FuzzySet[:] fuzzy_sets = np.empty(self.get_length(), dtype=object)
        cdef FuzzySet fs
        cdef int i

        for i in range(self.get_length()):
            fs = FuzzySet.wrap(cpp_fuzzy_sets[i])
            fuzzy_sets[i] = fs
        return fuzzy_sets

    cpdef get_support_values(self):
        """Get all the support values (one per fuzzy set) in an array
        
        Returns:
            float[]: Array of support values
        """
        return np.array(self.ptr.get_support_values(), dtype=np.float32)

    cpdef get_domain(self):
        """Get the domain of this variable (e.g [0,1])
        
        Returns:
            float[]: Domain of this variable (min and max values)
        """
        return np.array(self.ptr.get_domain(), dtype=np.float32)

    def get_plot(self, ax):
        """Draw the fuzzy variable fuzzy sets on the given matplotlib Axes object

        Args:
            ax (matplotlib.axes.Axes): Axes object

        Returns:
            matplotlib.axes.Axes: The axes object where we drew
        """
        cdef int i
        cdef FuzzySetCpp* fuzzy_set
        cdef vector[FuzzySetCpp*] cpp_fuzzy_sets = self.ptr.get_fuzzy_sets()
        cdef cnp.ndarray[float, ndim=2] points
        cdef float[:] domain = self.get_domain()

        ax.set_title(self.get_name())
        for i in range(self.get_length()):
            fuzzy_set = cpp_fuzzy_sets[i]
            points = np.array(fuzzy_set.get_function().get_plot_points(domain[0], domain[1]), dtype=np.float32)
            ax.plot(points[:,0], points[:,1], label=fuzzy_set.get_term().decode("utf-8"))

        ax.legend(loc="upper right")
        ax.set_xlim(domain)

        return ax

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return self.ptr.to_string().decode("utf-8")

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = FuzzyVariable.wrap(self.ptr)
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

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, FuzzyVariable):
            return False

        cdef FuzzyVariable other_c = <FuzzyVariable> other
        return self.ptr[0] == other_c.ptr[0]

    @staticmethod
    cdef FuzzyVariable wrap(FuzzyVariableCpp * wrapped_ptr):
        if wrapped_ptr == NULL:
            raise ValueError("pointer is NULL")
        cdef FuzzyVariable new_object = FuzzyVariable(None, do_init=False)
        new_object.ptr = wrapped_ptr.clone()
        return new_object