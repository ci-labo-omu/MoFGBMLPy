#ifndef ABSTRACT_ANTECEDENT_FACTORY_HPP
#define ABSTRACT_ANTECEDENT_FACTORY_HPP

#include <vector>
#include "../antecedent.hpp"
#include "../../../../utility/mersenne_twister_generator.hpp"

class AbstractAntecedentFactory {
protected:
    std::shared_ptr<Knowledge> knowledge;
    std::shared_ptr<RandomGenerator> random_generator;

public:
    AbstractAntecedentFactory(const std::shared_ptr<Knowledge>& knowledge,
                              const std::shared_ptr<RandomGenerator>& random_gen)
        : knowledge(knowledge), random_generator(random_gen) {
        if (!knowledge || knowledge->get_num_dim() == 0) {
            throw std::runtime_error("Knowledge is uninitialized or empty");
        }
    }
    virtual ~AbstractAntecedentFactory() {}

    virtual std::vector<std::shared_ptr<Antecedent>> create(int num_rules = 1) const = 0;
    virtual std::vector<std::vector<int>> create_antecedent_indices(int num_rules = 1) const = 0;
    virtual AbstractAntecedentFactory* clone() const = 0;

    virtual std::shared_ptr<Knowledge> get_knowledge() const {
        return knowledge;
    }

    virtual operator std::string() const = 0;
    std::string to_string() const {
        return static_cast<std::string>(*this);
    }
};

#endif
