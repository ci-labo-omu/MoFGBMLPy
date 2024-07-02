import numpy as np
import cython
from mofgbmlpy.fuzzy.fuzzy_term.dont_care_fuzzy_set import DontCareFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.linguistic_variable import LinguisticVariable
from mofgbmlpy.fuzzy.knowledge.abstract_knowledge_factory cimport AbstractKnowledgeFactory
from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory cimport HomoTriangleKnowledgeFactory
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
cimport numpy as cnp
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.triangular_fuzzy_set import TriangularFuzzySet


cdef class HomoTriangleKnowledgeFactory_2_3_4_5(HomoTriangleKnowledgeFactory):
    cpdef create(self)
