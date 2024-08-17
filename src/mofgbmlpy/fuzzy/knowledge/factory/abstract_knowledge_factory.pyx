cdef class AbstractKnowledgeFactory:
    """Abstract class for knowledge factories. It's used to create a Knowledge object"""
    cpdef create(self):
        """Create a Knowledge object
        
        Returns:
            Knowledge: Created knowledge
        """
        raise Exception("Abstract class")