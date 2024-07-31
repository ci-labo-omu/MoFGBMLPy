#include <deque>
#include <unordered_map>
#include <iostream>
#include "lru_cache.h"

using namespace std;


LRUCache::LRUCache(int maxSize) : maxSize(maxSize), dataMap(unordered_map<int,double>()), dataAccessHistory(deque<int>()) {
    this->dataMap.reserve(maxSize);
}

LRUCache::LRUCache() : LRUCache(128) {}

bool LRUCache::has(int key) {
    return this->dataMap.find(key) != this->dataMap.end();
}

double LRUCache::get(int key) {
    return this->dataMap.at(key);
}

void LRUCache::put(int key, double value) {
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
