#include <deque>
#include <unordered_map>
#include <iostream>
#include "lru_cache.h"

using namespace std;


LRUCache::LRUCache(int maxSize) : maxSize(maxSize), dataMap(unordered_map<int,double>()), dataAccessHistory(deque<int>()) {
    this->dataMap.reserve(maxSize);
}

LRUCache::LRUCache() : LRUCache(128) {}

inline int LRUCache::combine_keys(int key1, int key2) {
    return key1 ^ (key2 + 9152909 + (key1 << 5) + (key2 >> 3));
}

bool LRUCache::has(int key1, int key2) {
    return this->dataMap.find(combine_keys(key1, key2)) != this->dataMap.end();
}

double LRUCache::get(int key1, int key2) {
    return this->dataMap.at(combine_keys(key1, key2));
}

void LRUCache::put(int key1, int key2, double value) {
    int key = combine_keys(key1, key2);
    if(this->dataMap.size() >= this->maxSize) {
        //      Cache is full so remove the least recently used item (back)
        //      1. Update cache order
        int deleted_key = this->dataAccessHistory.back();
        this->dataAccessHistory.pop_back();
        //      2. Update cache content
        this->dataMap.erase(deleted_key);
    }

    this->dataMap[key] = value;
    this->dataAccessHistory.push_front(key);
}

int LRUCache::get_max_size() {
    return this->maxSize;
}

int LRUCache::get_size() {
    return this->dataMap.size();
}
