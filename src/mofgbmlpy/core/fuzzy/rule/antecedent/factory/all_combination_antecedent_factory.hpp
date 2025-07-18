#ifndef ALL_COMBINATION_ANTECEDENT_FACTORY_HPP
#define ALL_COMBINATION_ANTECEDENT_FACTORY_HPP

#include <vector>
#include <queue>
#include "../../../knowledge/knowledge.hpp"
#include "../antecedent.hpp"
#include "abstract_antecedent_factory.hpp"

class AllCombinationAntecedentFactory : public AbstractAntecedentFactory {
private:
    std::vector<std::vector<int>> antecedents_indices;

public:
    AllCombinationAntecedentFactory(const std::shared_ptr<Knowledge>& knowledge,
                              const std::shared_ptr<RandomGenerator>& random_gen);
    AllCombinationAntecedentFactory(const AllCombinationAntecedentFactory& other);

    int get_num_antecedents();
    std::vector<std::shared_ptr<Antecedent>> create(int num_rules = 1) const override;
    std::vector<std::vector<int>> create_antecedent_indices(int num_rules = 1) const override;
    AbstractAntecedentFactory* clone() const override;
    bool operator==(const AllCombinationAntecedentFactory& other) const;
    operator std::string() const override;

private:
    std::vector<std::vector<int>> generate_antecedents_indices() const;
};

#endif
