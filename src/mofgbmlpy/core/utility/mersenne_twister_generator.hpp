#ifndef RANDOM_GENERATOR_HPP
#define RANDOM_GENERATOR_HPP

#include <random>
#include <memory>

class RandomGenerator {
private:
    std::mt19937 generator;
    
public:
    // Constructor with seed
    explicit RandomGenerator(unsigned int seed = std::random_device{}());
    
    // Copy constructor
    RandomGenerator(const RandomGenerator& other);
    
    // Assignment operator
    RandomGenerator& operator=(const RandomGenerator& other);
    
    // Destructor
    ~RandomGenerator() = default;
    
    // Set seed
    void seed(unsigned int seed);
    
    // Generate random integer in range [min, max)
    int randint(int min, int max);
    
    // Generate random float in range [0.0, 1.0)
    double random();
    
    // Generate random float in range [min, max)
    double uniform(double min, double max);
    
    // Generate random float with normal distribution
    double normal(double mean = 0.0, double stddev = 1.0);
    
    // Get reference to underlying generator for compatibility
    std::mt19937& get_generator();
    const std::mt19937& get_generator() const;
    
    // Get current state for reproducibility
    std::string get_state() const;
    void set_state(const std::string& state);
    std::vector<int> generate_integers(int size, int pool_size, bool allow_replacement = true);
};

#endif
