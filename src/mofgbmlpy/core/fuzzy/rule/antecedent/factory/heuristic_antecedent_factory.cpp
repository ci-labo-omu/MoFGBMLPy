#include "heuristic_antecedent_factory.hpp"
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>

HeuristicAntecedentFactory::HeuristicAntecedentFactory(const std::shared_ptr<Dataset>& training_set,
                                                       const std::shared_ptr<Knowledge>& knowledge,
                                                       bool is_dc_probability,
                                                       double dc_rate,
                                                       int antecedent_number_do_not_dont_care,
                                                       const std::shared_ptr<RandomGenerator>& random_generator)
    : AbstractAntecedentFactory(knowledge, random_generator),
      training_set(training_set),
      is_dc_probability(is_dc_probability),
      dc_rate(dc_rate),
      antecedent_number_do_not_dont_care(antecedent_number_do_not_dont_care)
{
    if (!knowledge || knowledge->get_num_dim() == 0) {
        throw std::runtime_error("Knowledge is uninitialized or empty");
    }

    if (!training_set || training_set->get_size() == 0) {
        throw std::runtime_error("Training set is uninitialized or empty");
    }

    if (knowledge->get_num_dim() != training_set->get_num_dim()) {
        throw std::invalid_argument("Knowledge and training set dimensions do not match");
    }

    if (dc_rate < 0.0 || dc_rate > 1.0) {
        throw std::invalid_argument("dc_rate must be in the range [0.0, 1.0]");
    }

    if (antecedent_number_do_not_dont_care < 0) {
        throw std::invalid_argument("antecedent_number_do_not_dont_care must be non-negative");
    }

    if (!is_dc_probability)
        this->dc_rate = std::max(
            (knowledge->get_num_dim() - antecedent_number_do_not_dont_care) / static_cast<double>(knowledge->
                get_num_dim()), dc_rate);
}

HeuristicAntecedentFactory::HeuristicAntecedentFactory(const HeuristicAntecedentFactory& other)
    : HeuristicAntecedentFactory(other.training_set, other.knowledge,
                                 other.is_dc_probability, other.dc_rate,
                                 other.antecedent_number_do_not_dont_care,
                                 other.random_generator) {}

std::vector<int> HeuristicAntecedentFactory::select_antecedent_part(int index) const {
    auto pattern = training_set->get_pattern(index);
    return calculate_antecedent_part(*pattern);
}

std::vector<int> HeuristicAntecedentFactory::calculate_antecedent_part(const Pattern& pattern) const {
    if (pattern.get_num_dim() != knowledge->get_num_dim()) {
        throw std::invalid_argument("Pattern dimension mismatch");
    }

    int dim = knowledge->get_num_dim();
    std::vector<int> antecedent_indices(dim, 0);

    for (int i = 0; i < dim; ++i) {
        if (random_generator->random() < dc_rate) {
            // DC
            antecedent_indices[i] = 0;  // The first fuzzy set (index = 0) is don't care
            continue;
        }

        // Categorical judge
        double attr_val = pattern.get_attribute_value(i);
        if (attr_val < 0) {
            antecedent_indices[i] = static_cast<int>(attr_val);
            continue;
        }

        // Numerical (get a random fuzzy set index using the membership value)
        int num_fuzzy_sets = knowledge->get_num_fuzzy_sets(i) - 1;
        if (num_fuzzy_sets < 1) {
            antecedent_indices[i] = 0;  // don't care
            continue;
        }

        std::vector<double> mb_values_inc_sum(num_fuzzy_sets, 0.0);
        double sum_mb = 0.0;
        for (int h = 0; h < num_fuzzy_sets; ++h) {
            sum_mb += knowledge->get_membership_value(attr_val, i, h + 1);
            mb_values_inc_sum[h] = sum_mb;
        }

        double arrow = random_generator->random() * sum_mb;

        for (int h = 0; h < num_fuzzy_sets; ++h) {
            if (arrow < mb_values_inc_sum[h]) {
                antecedent_indices[i] = h + 1;
                break;
            }
        }
    }

    return antecedent_indices;
}

std::vector<std::shared_ptr<Antecedent>> HeuristicAntecedentFactory::create(int num_rules) const {
    if (num_rules <= 0) {
        throw std::invalid_argument("num_rules must be positive");
    }

    std::vector<std::vector<int>> indices = create_antecedent_indices(num_rules);
    std::vector<std::shared_ptr<Antecedent>> antecedents(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        antecedents[i] = std::make_shared<Antecedent>(indices[i], knowledge);
    }

    return antecedents;
}

std::vector<std::vector<int>> HeuristicAntecedentFactory::create_antecedent_indices_from_pattern(const Pattern& pattern) const {
    return { calculate_antecedent_part(pattern) };
}

std::vector<std::vector<int>> HeuristicAntecedentFactory::create_antecedent_indices(int num_rules) const {
    if (num_rules <= 0) {
        throw std::invalid_argument("num_rules must be positive");
    }

    int data_size = training_set->get_size();

    if (num_rules == 1) {
        return { select_antecedent_part(random_generator->randint(0, data_size)) };
    }

    std::vector<int> pattern_indices;
    pattern_indices.reserve(num_rules);

    if (num_rules <= data_size) {
        pattern_indices = random_generator->generate_integers(num_rules, data_size, false);
    } else {
        for (int i = 0; i < num_rules / data_size; i++) {
            for (int j = 0; j < data_size; j++) {
                pattern_indices.push_back(j);
            }
        }
        int num_remaining_indices = num_rules % data_size;

        std::vector<int> remaining_indices = random_generator->generate_integers(num_remaining_indices, data_size, false);

        for (int idx : remaining_indices) {
            pattern_indices.push_back(idx);
        }
    }

    std::vector<std::vector<int>> new_antecedent_indices;
    new_antecedent_indices.reserve(num_rules);

    for (int idx : pattern_indices) {
        new_antecedent_indices.push_back(select_antecedent_part(idx));
    }
    return new_antecedent_indices;
}

std::shared_ptr<Dataset> HeuristicAntecedentFactory::get_training_set() const {
    return training_set;
}

bool HeuristicAntecedentFactory::get_is_dc_probability() const {
    return is_dc_probability;
}

double HeuristicAntecedentFactory::get_dc_rate() const {
    return dc_rate;
}

int HeuristicAntecedentFactory::get_antecedent_number_do_not_dont_care() const {
    return antecedent_number_do_not_dont_care;
}

bool HeuristicAntecedentFactory::operator==(const HeuristicAntecedentFactory& other) const {
    return (training_set == other.get_training_set() &&
            *knowledge == *other.get_knowledge() &&
            is_dc_probability == other.get_is_dc_probability() &&
            dc_rate == other.get_dc_rate() &&
            antecedent_number_do_not_dont_care == other.get_antecedent_number_do_not_dont_care());
}

AbstractAntecedentFactory* HeuristicAntecedentFactory::clone() const {
    return new HeuristicAntecedentFactory(*this);
}

HeuristicAntecedentFactory::operator std::string() const
{
    return "HeuristicAntecedentFactory [training_set_size=" + std::to_string(training_set->get_size()) +
           ", dimension=" + std::to_string(knowledge->get_num_dim()) + "]";
}

