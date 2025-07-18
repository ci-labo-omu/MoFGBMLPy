#include "all_combination_antecedent_factory.hpp"

#include <iostream>
#include <limits>

AllCombinationAntecedentFactory::AllCombinationAntecedentFactory(const std::shared_ptr<Knowledge>& knowledge,
                              const std::shared_ptr<RandomGenerator>& random_gen) : AbstractAntecedentFactory(knowledge, random_gen) {
    antecedents_indices = generate_antecedents_indices();
}

AllCombinationAntecedentFactory::AllCombinationAntecedentFactory(const AllCombinationAntecedentFactory& other) : AbstractAntecedentFactory(other.knowledge, other.random_generator) {}

int AllCombinationAntecedentFactory::get_num_antecedents()
{
    return antecedents_indices.size();
}

std::vector<std::vector<int>> AllCombinationAntecedentFactory::generate_antecedents_indices() const {
    std::vector<std::vector<int>> indices;
    std::queue<std::vector<int>> indices_queue;
    std::vector<std::shared_ptr<FuzzyVariable>> fuzzy_vars = knowledge->get_fuzzy_vars();
    int num_dim = knowledge->get_num_dim();
    int num_generated_indices = 1;

    for (int i = 0; i < num_dim; ++i) {
        FuzzyVariable* fuzzy_var = fuzzy_vars[i].get();
        int var_length = fuzzy_var->get_length();

        if (num_generated_indices > INT_MAX / var_length) {
            std::cout << "WARNING: Too many antecedent indices to be generated, not all combinations will be generated" << std::endl;
            num_generated_indices = INT_MAX;
            break;
        }
        num_generated_indices *= var_length;
    }

    try {
        indices.reserve(num_generated_indices);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("The number of variables and/or the number of fuzzy sets is too big, the antecedents list memory can't be allocated. Please use another antecedent factory");
    }

    indices_queue.push(std::vector<int>());

    // Generate all combinations of fuzzy sets indices
    int k = 0;
    while (!indices_queue.empty()) {
        std::vector<int> buffer = indices_queue.front();
        indices_queue.pop();
        int current_dim = buffer.size();
        if (current_dim < num_dim) {
            const FuzzyVariable* var = fuzzy_vars[current_dim].get();
            for (int i = 0; i < var->get_length(); i++) {
                std::vector<int> tmp(buffer);
                tmp.push_back(i);
                indices_queue.push(tmp);
            }
        } else {
            // A list of antecedent indices is full so we can add it
            indices.push_back(buffer);
            k++;
            if (k >= num_generated_indices) {
                break;
            }
        }
    }

    return indices;
}

std::vector<std::shared_ptr<Antecedent>> AllCombinationAntecedentFactory::create(int num_rules) const {
    std::vector<std::vector<int>> antecedents_indices = create_antecedent_indices(num_rules);
    std::vector<std::shared_ptr<Antecedent>> antecedents = std::vector<std::shared_ptr<Antecedent>>(num_rules);

    for (int i = 0; i < num_rules && i < antecedents_indices.size(); ++i) {
        antecedents[i] = std::make_shared<Antecedent>(antecedents_indices[i], knowledge);
    }

    return antecedents;
}

std::vector<std::vector<int>> AllCombinationAntecedentFactory::create_antecedent_indices(int num_rules) const {
    std::vector<std::vector<int>> indices(num_rules, std::vector<int>(knowledge->get_num_dim()));

    if (num_rules <= 0) {
        throw std::invalid_argument("num_rules must be positive");
    }

    num_rules = std::min(num_rules, static_cast<int>(antecedents_indices.size()));

    std::vector<int> const chosen_indices_lists = random_generator->generate_integers(num_rules, antecedents_indices.size(), false);

    for (int i = 0; i < num_rules; ++i) {
        const std::vector<int>& chosen_list = antecedents_indices[chosen_indices_lists[i]];
        for (int j = 0; j < chosen_list.size(); ++j) {
            indices[i][j] = chosen_list[j];
        }
    }
    return indices;
}

AbstractAntecedentFactory* AllCombinationAntecedentFactory::clone() const {
    return new AllCombinationAntecedentFactory(*this);
}

bool AllCombinationAntecedentFactory::operator==(const AllCombinationAntecedentFactory& other) const {
    return *knowledge == *other.knowledge;
}

AllCombinationAntecedentFactory::operator std::string() const
{
    return "AllCombinationAntecedentFactory [antecedents=" + std::to_string(antecedents_indices.size()) + ", dimension=" + std::to_string(knowledge->get_num_dim()) + "]";
}

