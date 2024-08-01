import xml.etree.cElementTree as xml_tree
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType


cdef class FuzzySet:
    def __init__(self, AbstractMF function, int id, int division_type, str term=""):
        if function is None or id is None or division_type is None or term is None:
            raise ValueError("function, id, division_type and term can't be none")

        if id < 0 or division_type < 0 or division_type >= len(DivisionType):
            raise ValueError("Invalid DivisionType constant value")

        self.__function = function
        self.__term = term
        self.__id = id
        self.__division_type = division_type

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"Fuzzy set {self.__term}"

    cdef double get_membership_value(self, double x):
        return self.__function.get_value(x)

    cpdef get_term(self):
        return self.__term

    cpdef get_function_callable(self):
        return self.__function.get_value

    cpdef int get_id(self):
        return self.__id

    cpdef AbstractMF get_function(self):
        return self.__function

    cpdef get_division_type(self):
        return self.__division_type

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("fuzzyTerm")
        term_xml = xml_tree.SubElement(root, "fuzzyTermID")
        term_xml.text = str(self.get_id())

        term_xml = xml_tree.SubElement(root, "fuzzyTermName")
        term_xml.text = self.get_term()

        term_xml = xml_tree.SubElement(root, "ShapeTypeName")
        term_xml.text = str(self.__function.__class__.__name__)

        root.append(self.__function.to_xml())

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

        return (self.__id == other.get_id() and
                self.__function == other.get_function() and
                self.__term == other.get_term()
                and self.__division_type == other.get_division_type())
