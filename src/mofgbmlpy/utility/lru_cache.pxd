cdef extern from "lru_cache.cpp":
    pass

# Declare the class with cdef
cdef extern from "lru_cache.h":
    cdef cppclass LRUCache:
        LRUCache(int maxSize)
        LRUCache()
        bint has(int key)
        double get(int key)
        void put(int key, double value)
        int get_max_size()
        int get_size()