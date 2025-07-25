#include "homo_triangle_knowledge_factory_2_3_4_5.hpp"

HomoTriangleKnowledgeFactory_2_3_4_5::HomoTriangleKnowledgeFactory_2_3_4_5(int num_dims, const std::vector<std::string>& var_names)
    : HomoTriangleKnowledgeFactory(create_num_divisions(num_dims),
        var_names.empty() ? get_fuzzy_set_names(num_dims) : var_names,
        create_fuzzy_set_names(num_dims)) {

    if (num_dims <= 0)
    {
        throw std::runtime_error("num_dims must be positive");
    }
}

std::vector<std::vector<int>> HomoTriangleKnowledgeFactory_2_3_4_5::create_num_divisions(int num_dims) {
    std::vector<int> divisions = {2, 3, 4, 5};
    std::vector<std::vector<int>> result(num_dims, divisions);
    return result;
}

std::vector<std::vector<std::vector<std::string>>> HomoTriangleKnowledgeFactory_2_3_4_5::create_fuzzy_set_names(int num_dims) {
    std::vector<std::vector<std::string>> labels = {
        {"low_2", "high_2"},
        {"low_3", "medium_3", "high_3"},
        {"low_4", "low_medium_4", "high_medium_4", "high_4"},
        {"very_low_5", "low_5", "medium_5", "high_5", "very_high_5"}
    };
    std::vector<std::vector<std::vector<std::string>>> result(num_dims, labels);
    return result;
}
