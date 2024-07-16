import xml.etree.cElementTree as xml_tree

cdef class FuzzySet:
    def __init__(self, function, id, division_type, term=""):
        self.__function = function
        self.__term = term
        self.__id = id
        self.__division_type = division_type

    def __repr__(self):
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
        root = xml_tree.Element("fuzzyTerm")
        term_xml = xml_tree.SubElement(root, "fuzzyTermID")
        term_xml.text = str(self.get_id())

        term_xml = xml_tree.SubElement(root, "fuzzyTermName")
        term_xml.text = self.get_term()

        term_xml = xml_tree.SubElement(root, "ShapeTypeName")
        term_xml.text = str(self.__function.__class__.__name__)

        root.append(self.__function.to_xml())

        return root
