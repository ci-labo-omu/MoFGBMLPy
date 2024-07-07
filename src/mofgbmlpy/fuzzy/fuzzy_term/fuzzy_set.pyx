import xml.etree.cElementTree as xml_tree

cdef class FuzzySet:
    def __init__(self, function, id, term=""):
        self.__function = function
        self.__term = term
        self.__id = id

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

    #     <fuzzySets dimension = "0" >
    #         <fuzzyTerm>
        #         <fuzzyTermID> 0 < / fuzzyTermID >
        #         <fuzzyTermName> rectangularShape_equalDivision_99 < / fuzzyTermName >
        #         <ShapeTypeID> 9 < / ShapeTypeID >
        #         <ShapeTypeName> rectangularShape < / ShapeTypeName >
        #         <divisionType> equalDivision < / divisionType >
        #         <partitionNum> 0 < / partitionNum >
        #         <partition_i> 0 < / partition_i >
        #         <parameterSet>
            #         <parameter id = "0" > 0.0 < / parameter >
            #         <parameter id = "1" > 1.0 < / parameter >
            #     </ parameterSet>
        # </ fuzzyTerm>
