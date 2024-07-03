import xml.etree.cElementTree as xml_tree

cdef class FuzzySet:
    def __init__(self, function, term=""):
        self.__function = function
        self.__term = term

    def __repr__(self):
        return f"Fuzzy set {self.__term}"

    cdef double get_membership_value(self, double x):
        return self.__function.get_value(x)

    cpdef get_term(self):
        return self.__term

    cpdef get_function_callable(self):
        return self.__function.get_value

    def to_xml(self):
        root = xml_tree.Element("fuzzy-set")
        term_xml = xml_tree.SubElement(root, "term")
        term_xml.text = self.__term

        root.append(self.__function.to_xml())

        return root