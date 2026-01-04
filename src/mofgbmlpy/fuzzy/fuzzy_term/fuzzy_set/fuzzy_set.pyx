import copy
import xml.etree.cElementTree as xml_tree
from mofgbmlpy.fuzzy.fuzzy_term.membership_function.abstract_mf cimport AbstractMF
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.division_type import DivisionType
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.rectangular_fuzzy_set import RectangularFuzzySet

cdef class FuzzySet:
    """Fuzzy set

    Attributes:
        _function (AbstractMF): Membership function
        _term (str): Name of the fuzzy set (e.g. small)
        _id (int): ID of the fuzzy set
        _division_type (int): Division type of this fuzzy set (e.g. EQUAL_DIVISION)
    """
    def __init__(self, AbstractMF function, int id, int division_type, str term=""):
        if function is None or id is None or division_type is None or term is None:
            raise TypeError("function, id, division_type and term can't be none")

        if id < 0 or division_type < 0 or division_type >= len(DivisionType):
            raise ValueError("Invalid DivisionType constant value")

        self._function = function
        self._term = term
        self._id = id
        self._division_type = division_type  # TODO: not yet implemented

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return f"Fuzzy set {self._term}"

    cdef float get_membership_value(self, float x):
        """Get the membership value of a value for this fuzzy set
        
        Args:
            x (float): Value whose membership value is calculated

        Returns:
            float: Membership value
        """
        return self._function.get_value(x)

    cpdef get_term(self):
        """Get the name associated to the fuzzy set
        
        Returns:
            str: Name associated to the fuzzy set
        """
        return self._term

    cpdef get_function_callable(self):
        """Get the membership function object's function
        
        Returns:
            function: Membership function
        """
        return self._function.get_value

    cpdef int get_id(self):
        """Get th ID of Fuzzy set
                
        Returns:
            int: Fuzzy set ID
        """
        return self._id

    cpdef AbstractMF get_function(self):
        """Get the membership function object
        
        Returns:
            AbstractMF: Membership function object
        """
        return self._function

    cpdef set_function(self, AbstractMF function):
        """Set the membership function object
        
        Args:
            function (AbstractMF): Membership function object
        """
        if function is None:
            raise TypeError("function can't be none")
        self._function = function

    cpdef get_division_type(self):
        """Get the division type of this fuzzy set
        
        Returns:
            DivisionType: Division type

        """
        return self._division_type

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
        term_xml.text = str(self._function.__class__.__name__)

        root.append(self._function.to_xml())

        return root

    @staticmethod
    def from_xml(xml_element):
        """Create a FuzzySet object from its XML representation

        Args:
            xml_element (xml.etree.ElementTree): XML element representing the fuzzy set

        Returns:
            FuzzySet: FuzzySet object created from the XML element
        """
        cdef int fuzzy_set_id = int(xml_element.find("fuzzyTermID").text)
        cdef str fuzzy_set_term = xml_element.find("fuzzyTermName").text
        cdef str shape_type_name = xml_element.find("ShapeTypeName").text
        cdef str fs_type_name = shape_type_name.replace(r'MF', 'FuzzySet').replace(r'Shape', 'FuzzySet')
        fs_type_name = fs_type_name[0].upper() + fs_type_name[1:]

        param_set = xml_element.find("parameterSet")
        new_params = []
        if param_set is not None:
            imported_params = xml_element.find("parameterSet").findall("parameter")
            for param in imported_params:
                new_params.append(float(param.text))

            if fs_type_name == "RectangularFuzzySet" and new_params[0] == 0.0 and new_params[1] == 1.0:
                # In the Java version DC is saved as rectangular
                fs_type_name = "DontCareFuzzySet"
            else:
                new_params += [fuzzy_set_id, fuzzy_set_term]
        if fs_type_name == "DontCareFuzzySet":
            new_params = [fuzzy_set_id]

        cdef FuzzySet new_fuzzy_set = globals()[fs_type_name](*new_params)

        return new_fuzzy_set

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, FuzzySet):
            return False

        return (self._id == other.get_id() and
                self._function == other.get_function() and
                self._term == other.get_term()
                and self._division_type == other.get_division_type())

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = FuzzySet(copy.deepcopy(self._function, memo),
                                self._id,
                                self._division_type,
                                self._term)
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
        return self._function.get_support(x_min, x_max)