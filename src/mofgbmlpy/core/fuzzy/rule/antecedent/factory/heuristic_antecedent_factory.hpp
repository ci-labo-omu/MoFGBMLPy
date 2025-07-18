#ifndef HEURISTIC_ANTECEDENT_FACTORY_HPP
#define HEURISTIC_ANTECEDENT_FACTORY_HPP

#include <vector>
#include <random>
#include "../../../../data/dataset.hpp"
#include "../../../knowledge/knowledge.hpp"
#include "../antecedent.hpp"
#include "../../../../data/pattern.hpp"
#include "abstract_antecedent_factory.hpp"
#include "../../../../utility/mersenne_twister_generator.hpp"

class HeuristicAntecedentFactory : public AbstractAntecedentFactory {
private:
    std::shared_ptr<Dataset> training_set;
    bool is_dc_probability;
    double dc_rate;
    int antecedent_number_do_not_dont_care;

    std::vector<int> select_antecedent_part(int index) const;

public:
    HeuristicAntecedentFactory(const std::shared_ptr<Dataset>& training_set,
                               const std::shared_ptr<Knowledge>& knowledge,
                               bool is_dc_probability, double dc_rate, int antecedent_number_do_not_dont_care,
                               const std::shared_ptr<RandomGenerator>& random_gen);
    HeuristicAntecedentFactory(const HeuristicAntecedentFactory& other);

    std::vector<int> calculate_antecedent_part(const Pattern& pattern) const;
    std::vector<std::shared_ptr<Antecedent>> create(int num_rules=1) const override;
    std::vector<std::vector<int>> create_antecedent_indices_from_pattern(const Pattern& pattern) const;
    std::vector<std::vector<int>> create_antecedent_indices(int num_rules=1) const override;
    std::shared_ptr<Dataset> get_training_set() const;
    bool get_is_dc_probability() const;
    double get_dc_rate() const;
    int get_antecedent_number_do_not_dont_care() const;

    bool operator==(const HeuristicAntecedentFactory& other) const;
    AbstractAntecedentFactory* clone() const override;
    operator std::string() const override;
};

#endif
