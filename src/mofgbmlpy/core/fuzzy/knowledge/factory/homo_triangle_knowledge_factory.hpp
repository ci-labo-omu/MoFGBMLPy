#ifndef HOMO_TRIANGLE_KNOWLEDGE_FACTORY_HPP
#define HOMO_TRIANGLE_KNOWLEDGE_FACTORY_HPP

#include <vector>
#include <string>
#include <memory>

#include "../knowledge.hpp"  // Your Knowledge C++ class
#include "../../fuzzy_term/fuzzy_variable.hpp"
#include "../../fuzzy_term/fuzzy_set/fuzzy_set.hpp"
#include "../../fuzzy_term/fuzzy_set/dont_care_fuzzy_set.hpp"
#include "../../fuzzy_term/fuzzy_set/triangular_fuzzy_set.hpp"

class HomoTriangleKnowledgeFactory {
private:
    std::vector<std::vector<int>> num_divisions;
    std::vector<std::string> var_names;
    std::vector<std::vector<std::vector<std::string>>> fuzzy_set_names;

    static std::vector<std::vector<float>> make_triangle_knowledge_params(int num_fuzzy_sets);

protected:
    static std::vector<std::string> get_fuzzy_set_names(int num_dims);

public:
    HomoTriangleKnowledgeFactory(const std::vector<std::vector<int>>& num_divisions,
                                 const std::vector<std::string>& var_names,
                                 const std::vector<std::vector<std::vector<std::string>>>& fuzzy_set_names);

    HomoTriangleKnowledgeFactory(const HomoTriangleKnowledgeFactory& other);

    std::shared_ptr<Knowledge> create();
    HomoTriangleKnowledgeFactory* clone() const;
};

#endif
