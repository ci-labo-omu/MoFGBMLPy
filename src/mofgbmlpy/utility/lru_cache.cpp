#include "lru_cache.h"

using namespace std;


LRUCache::LRUCache(int maxSize) : dataMap(unordered_map<size_t,double>()), queueIteratorsMap(unordered_map<size_t, list<size_t>::iterator>()), dataAccessHistory(list<size_t>()), maxSize(maxSize) {}

LRUCache::LRUCache() : LRUCache(128) {}

size_t LRUCache::combine_keys(int key1, int key2) {
    return ((key1+1 << 16)+ key2); // key1 and key2 are 32bits so if each one take 32 bits of the 64bits we have few collisions
}


bool LRUCache::has(int key1, int key2) {
    return this->dataMap.find(combine_keys(key1, key2)) != this->dataMap.end();
}

double LRUCache::get(int key1, int key2) {
    size_t key = combine_keys(key1, key2);
    update_last_access(key);
    return this->dataMap.at(key);
}

void LRUCache::update_last_access(size_t key) {
    // Move the key to the beginning of the list
    dataAccessHistory.splice(dataAccessHistory.begin(), dataAccessHistory, queueIteratorsMap[key]);
}

void LRUCache::put(int key1, int key2, double value) {
    // Combine the two keys
    size_t key = combine_keys(key1, key2);
    auto element_iterator = dataMap.find(key);

    if (element_iterator != dataMap.end()) {
        // It already exists
        update_last_access(key);
        dataMap[key] = value;
    }

    if(this->dataMap.size() >= this->maxSize) {
        // Cache is full so remove the least recently used item (back)
        size_t deleted_key = dataAccessHistory.back();

        queueIteratorsMap.erase(deleted_key);
        dataAccessHistory.pop_back();

        dataMap.erase(deleted_key);
    }

    dataAccessHistory.push_front(key);
    queueIteratorsMap[key] = dataAccessHistory.begin();
    dataMap[key] = value;
}

int LRUCache::get_max_size() {
    return this->maxSize;
}


int LRUCache::get_size()
{
    return this->dataMap.size();
}
