#include "mersenne_twister_generator.hpp"

#include <set>
#include <sstream>
#include <stdexcept>

RandomGenerator::RandomGenerator(unsigned int seed) : generator(seed) {
}

RandomGenerator::RandomGenerator(const RandomGenerator& other)
    : generator(other.generator) {
}

RandomGenerator& RandomGenerator::operator=(const RandomGenerator& other) {
    if (this != &other) {
        generator = other.generator;
    }
    return *this;
}

void RandomGenerator::seed(unsigned int seed) {
    generator.seed(seed);
}

int RandomGenerator::randint(int min, int max) {
    if (min >= max) {
        throw std::invalid_argument("min must be less than max");
    }
    std::uniform_int_distribution<int> dist(min, max - 1);
    return dist(generator);
}

double RandomGenerator::random() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(generator);
}

double RandomGenerator::uniform(double min, double max) {
    if (min >= max) {
        throw std::invalid_argument("min must be less than max");
    }
    std::uniform_real_distribution<double> dist(min, max);
    return dist(generator);
}

double RandomGenerator::normal(double mean, double stddev) {
    if (stddev <= 0.0) {
        throw std::invalid_argument("stddev must be positive");
    }
    std::normal_distribution<double> dist(mean, stddev);
    return dist(generator);
}

std::mt19937& RandomGenerator::get_generator() {
    return generator;
}

const std::mt19937& RandomGenerator::get_generator() const {
    return generator;
}

std::string RandomGenerator::get_state() const {
    std::ostringstream oss;
    oss << generator;
    return oss.str();
}

void RandomGenerator::set_state(const std::string& state) {
    std::istringstream iss(state);
    iss >> generator;
    if (iss.fail()) {
        throw std::invalid_argument("Invalid state string");
    }
}

std::vector<int> RandomGenerator::generate_integers(int size, int pool_size, bool allow_replacement) {
    if (size < 0) {
        throw std::invalid_argument("size must be non-negative");
    }

    if (!allow_replacement && size > pool_size) {
        throw std::invalid_argument("size must be less than or equal to max if replacement is not allowed");
    }

    std::vector<int> result;
    result.reserve(size);

    if (allow_replacement) {
        std::uniform_int_distribution<int> dist(0, pool_size - 1);
        for (int i = 0; i < size; ++i) {
            result.push_back(dist(generator));
        }
    } else {
        std::vector<int> pool(pool_size);
        for (int i = 0; i < pool_size; ++i) {
            pool[i] = i;
        }
        for (int i = 0; i < size; ++i) {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(pool.size()) - 1);
            int idx = dist(generator);
            result.push_back(pool[idx]);
            pool.erase(pool.begin() + idx);
        }
    }
    return result;
}

