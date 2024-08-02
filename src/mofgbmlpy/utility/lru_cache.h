#ifndef LRU_CACHE_H
#define LRU_CACHE_H

#include <deque>
#include <unordered_map>

using namespace std;

class LRUCache {
private:
    unordered_map<size_t, double> dataMap;
    unordered_map<size_t, list<size_t>::iterator> queueIteratorsMap;
    list<size_t> dataAccessHistory;
    int maxSize;

public:
    LRUCache(int maxSize);
    LRUCache();
    inline static size_t combine_keys(int key1, int key2);
    bool has(int key1, int key2);
    double get(int key1, int key2);
    void update_last_access(size_t key);
    void put(int key1, int key2, double value);
    int get_max_size();
    int get_size();
};

#endif
