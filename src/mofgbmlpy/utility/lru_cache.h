#ifndef LRU_CACHE_H
#define LRU_CACHE_H

#include <deque>
#include <unordered_map>

using namespace std;

class LRUCache {
private:
    unordered_map<int, double> dataMap;
    deque<int> dataAccessHistory;
    int maxSize;

public:
    LRUCache(int maxSize);
    LRUCache();
    inline int combine_keys(int key1, int key2);
    bool has(int key1, int key2);
    double get(int key1, int key2);
    void put(int key1, int key2, double value);
    int get_max_size();
    int get_size();
};

#endif
