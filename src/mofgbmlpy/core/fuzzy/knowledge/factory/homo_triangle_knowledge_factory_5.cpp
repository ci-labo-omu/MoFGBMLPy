#include "homo_triangle_knowledge_factory_5.hpp"

HomoTriangleKnowledgeFactory_5::HomoTriangleKnowledgeFactory_5(int num_dims, const std::vector<std::string>& var_names)
    : HomoTriangleKnowledgeFactory(create_num_divisions(num_dims),
        var_names.empty() ? get_fuzzy_set_names(num_dims) : var_names,
        create_fuzzy_set_names(num_dims)) {}

std::vector<std::vector<int>> HomoTriangleKnowledgeFactory_5::create_num_divisions(int num_dims) {
    std::vector<int> divisions = {5};
    std::vector<std::vector<int>> result(num_dims, divisions);
    return result;
}

std::vector<std::vector<std::vector<std::string>>> HomoTriangleKnowledgeFactory_5::create_fuzzy_set_names(int num_dims) {
    std::vector<std::vector<std::string>> labels = {
        {"very_low_5", "low_5", "medium_5", "high_5", "very_high_5"}
    };
    std::vector<std::vector<std::vector<std::string>>> result(num_dims, labels);
    return result;
}
