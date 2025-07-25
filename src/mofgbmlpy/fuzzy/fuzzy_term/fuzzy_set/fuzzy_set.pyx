import copy
import xml.etree.cElementTree as xml_tree
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType

from mofgbmlpy.fuzzy.fuzzy_term.membership_function.mf_wrapper cimport wrap_mf

cdef class FuzzySet:
    """Fuzzy set

    Attributes:
        __function (AbstractMF): Membership function
        __term (str): Name of the fuzzy set (e.g. small)
        __id (int): ID of the fuzzy set
        __division_type (int): Division type of this fuzzy set (e.g. EQUAL_DIVISION)
    """
    # def __cinit__(self, AbstractMF function, int id, int division_type, str term="", do_init=True):
    #     if not do_init:
    #         self.ptr = NULL
    #         return
    #
    #     if function is None:
    #         raise ValueError("Membership function cannot be None")
    #
    #     if term is None:
    #         raise TypeError("Term cannot be None")
    #
    #     cdef DivisionTypeCpp cpp_div_type = <DivisionTypeCpp><int>division_type
    #
    #     self.ptr = new FuzzySetCpp(function.ptr.clone(), id, cpp_div_type, term.encode("utf-8"))

    def __cinit(self):
        """Constructor"""
        self.ptr = NULL

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return self.ptr.to_string().decode("utf-8")

    cdef float get_membership_value(self, float x):
        """Get the membership value of a value for this fuzzy set
        
        Args:
            x (float): Value whose membership value is calculated

        Returns:
            float: Membership value
        """
        return self.ptr.get_membership_value(x)

    cpdef get_term(self):
        """Get the name associated to the fuzzy set
        
        Returns:
            str: Name associated to the fuzzy set
        """
        return self.ptr.get_term().decode("utf-8")

    cpdef get_function_callable(self):
        """Get the membership function object's function
        
        Returns:
            function: Membership function
        """
        return self.get_function().get_value

    cpdef int get_id(self):
        """Get th ID of Fuzzy set
                
        Returns:
            int: Fuzzy set ID
        """
        return self.ptr.get_id()

    cpdef AbstractMF get_function(self):
        """Get the membership function object
        
        Returns:
            AbstractMF: Membership function object
        """
        return wrap_mf(self.ptr.get_function())

    cpdef set_function(self, AbstractMF function):
        """Set the membership function object
        
        Args:
            function (AbstractMF): Membership function object
        """
        self.ptr.set_function(function.ptr.clone())

    cpdef get_division_type(self):
        """Get the division type of this fuzzy set
        
        Returns:
            DivisionType: Division type

        """
        return DivisionType(self.ptr.get_division_type())

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        cdef AbstractMF function = self.get_function()

        root = xml_tree.Element("fuzzyTerm")
        term_xml = xml_tree.SubElement(root, "fuzzyTermID")
        term_xml.text = str(self.get_id())

        term_xml = xml_tree.SubElement(root, "fuzzyTermName")
        term_xml.text = self.get_term()

        term_xml = xml_tree.SubElement(root, "ShapeTypeName")
        term_xml.text = str(function.__class__.__name__)

        root.append(function.to_xml())

        return root

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, FuzzySet):
            return False

        cdef FuzzySet other_set = <FuzzySet>other

        return self.ptr[0] == other_set.ptr[0]

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = FuzzySet.wrap(self.ptr)
        memo[id(self)] = new_object
        return new_object

    cpdef float get_support(self, float x_min=0, float x_max=0):
        """Get the support value associated to this function: area covered by this function in the space "domain x [0, 1]"

        Args:
            x_min (float): Min value of the domain for the x axis
            x_max (float): Max value of the domain for the x axis

        Returns:
            Support value
        """
        return self.ptr.get_support(x_min, x_max)

    @staticmethod
    cdef FuzzySet wrap(FuzzySetCpp * wrapped_ptr):
        if wrapped_ptr == NULL:
            raise ValueError("pointer is NULL")

        cdef FuzzySet new_object = FuzzySet(function=None, id=0, division_type=0, do_init=False)
        new_object.ptr = wrapped_ptr.clone()
        return new_object
