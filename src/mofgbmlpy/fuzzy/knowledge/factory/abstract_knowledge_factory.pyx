cdef class AbstractKnowledgeFactory:
    cpdef create(self):
        raise Exception("Abstract class")