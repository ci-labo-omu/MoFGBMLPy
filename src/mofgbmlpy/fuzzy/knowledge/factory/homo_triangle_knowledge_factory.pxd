from mofgbmlpy.fuzzy.knowledge.factory.abstract_knowledge_factory cimport AbstractKnowledgeFactory

cdef class HomoTriangleKnowledgeFactory(AbstractKnowledgeFactory):
    cdef int[:,:] __num_divisions
    cdef object __var_names
    cdef object __fuzzy_set_names

    cpdef create(self)