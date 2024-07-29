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
    bool has(int key);
    double get(int key);
    void put(int key, double value);
    int get_max_size();
    int get_size();
};

#endif
