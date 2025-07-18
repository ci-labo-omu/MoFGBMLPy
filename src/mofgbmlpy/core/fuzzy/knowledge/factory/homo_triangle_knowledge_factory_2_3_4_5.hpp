#ifndef HOMO_TRIANGLE_KNOWLEDGE_FACTORY_2_3_4_5_HPP
#define HOMO_TRIANGLE_KNOWLEDGE_FACTORY_2_3_4_5_HPP

#include <string>
#include <vector>
#include <memory>
#include "homo_triangle_knowledge_factory.hpp"

class HomoTriangleKnowledgeFactory_2_3_4_5 : public HomoTriangleKnowledgeFactory {
    static std::vector<std::vector<int>> create_num_divisions(int num_dims);
    static std::vector<std::vector<std::vector<std::string>>> create_fuzzy_set_names(int num_dims);

public:
    HomoTriangleKnowledgeFactory_2_3_4_5(int num_dims, const std::vector<std::string>& var_names);
};

#endif
