cdef extern from "lru_cache.cpp":
    pass

# Declare the class with cdef
cdef extern from "lru_cache.h":
    cdef cppclass LRUCache:
        LRUCache(int maxSize)
        LRUCache()
        size_t combine_keys(int key1, int key2);
        bint has(int key1, int key2)
        double get(int key1, int key2)
        void put(int key1, int key2, double value)
        int get_max_size()
        int get_size()