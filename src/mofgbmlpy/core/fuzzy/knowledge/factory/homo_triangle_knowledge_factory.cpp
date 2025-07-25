#include "homo_triangle_knowledge_factory.hpp"
#include <stdexcept>

HomoTriangleKnowledgeFactory::HomoTriangleKnowledgeFactory(
    const std::vector<std::vector<int>>& num_divisions,
    const std::vector<std::string>& var_names,
    const std::vector<std::vector<std::vector<std::string>>>& fuzzy_set_names)
    : num_divisions(num_divisions), var_names(var_names), fuzzy_set_names(fuzzy_set_names)
{
    if (num_divisions.empty() || var_names.empty() || fuzzy_set_names.empty()) {
        throw std::invalid_argument("Parameters can't be empty");
    }

    if (num_divisions.empty() || num_divisions[0].size() == 0)
    {
        throw std::invalid_argument("num_divisions second dimension can't be null");
    }

    int num_dims = static_cast<int>(num_divisions.size());

    if (static_cast<int>(var_names.size()) != num_dims || static_cast<int>(fuzzy_set_names.size()) != num_dims) {
        throw std::invalid_argument("var_names and num_divisions first dimension must be of the same size as num_divisions first one");
    }

    for (const auto& item : var_names) {
        if (item.empty()) {
            throw std::invalid_argument("Var names (var_names items) can't be empty");
        }
    }

    for (int dim_i = 0; dim_i < num_dims; dim_i++) {
        int num_divisions_dim_i = static_cast<int>(num_divisions[dim_i].size());
        if (static_cast<int>(fuzzy_set_names[dim_i].size()) != num_divisions_dim_i) {
            throw std::invalid_argument("fuzzy_set_names second dimension is invalid");
        }
        for (int j = 0; j < num_divisions_dim_i; ++j) {
            if (static_cast<int>(fuzzy_set_names[dim_i][j].size()) != num_divisions[dim_i][j]) {
                throw std::invalid_argument("fuzzy_set_names third dimension is invalid");
            }
            if (num_divisions[dim_i][j] <= 0) {
                throw std::invalid_argument("num_divisions can't contain null or negative values");
            }
        }
    }
}

HomoTriangleKnowledgeFactory::HomoTriangleKnowledgeFactory(const HomoTriangleKnowledgeFactory& other) :
    num_divisions(other.num_divisions),
    var_names(other.var_names),
    fuzzy_set_names(other.fuzzy_set_names)
{}

std::vector<std::vector<float>> HomoTriangleKnowledgeFactory::make_triangle_knowledge_params(int num_fuzzy_sets) {
    if (num_fuzzy_sets <= 1) {
        throw std::invalid_argument("num_fuzzy_sets must be > 1");
    }

    std::vector<std::vector<float>> params(num_fuzzy_sets, std::vector<float>(3, 0.f));
    std::vector<float> partition(num_fuzzy_sets + 1, 0.f);

    // # e.g.: K = 2: 0, 1/2, 1
    // # e.g.: K = 3: 0, 1/4, 3/4, 1
    // # e.g.: K = 5: 0, 1/8, 3/8, 5/8, 7/8, 1

    for (int i = 1; i < num_fuzzy_sets; i++) {
        partition[i] = static_cast<float>(2 * i - 1) / ((num_fuzzy_sets - 1) * 2);
    }
    partition[num_fuzzy_sets] = 1.f;

    for (int i = 0; i < num_fuzzy_sets; i++) {
        if (i == 0) { // 1st partition
            params[i] = {0.f, 0.f, 2 * partition[1]};
        } else if (i == static_cast<int>(partition.size()) - 2) { // last partition
            params[i] = {2 * partition[i] - 1, 1.f, 1.f};
        } else if (i>0 && i < static_cast<int>(partition.size())-2) { // if the index is valid
            float left = partition[i] * 1.5f - partition[i + 1] * 0.5f;
            float center = (partition[i] + partition[i + 1]) / 2.f;
            float right = partition[i + 1] * 1.5f - partition[i] * 0.5f;
            params[i] = {left, center, right};
        }
    }

    return params;
}

Knowledge* HomoTriangleKnowledgeFactory::create() {
    auto knowledge = new Knowledge();

    int set_id = 0;
    std::vector<FuzzyVariable*> fuzzy_vars = std::vector<FuzzyVariable*>(num_divisions.size());

    for (int dim_i = 0; dim_i < static_cast<int>(num_divisions.size()); dim_i++) {
        std::vector<FuzzySet*> current_set;
        current_set.push_back(new DontCareFuzzySet(set_id));
        set_id++;

        for (int j = 0; j < static_cast<int>(num_divisions[dim_i].size()); j++) {
            int partition_size = num_divisions[dim_i][j];
            auto params = make_triangle_knowledge_params(partition_size);
            for (int div_i = 0; div_i < partition_size; div_i++) {
                float left = params[div_i][0];
                float center = params[div_i][1];
                float right = params[div_i][2];

                auto fuzzy_set = new TriangularFuzzySet(left, center, right, set_id, fuzzy_set_names[dim_i][j][div_i]);
                set_id++;
                current_set.push_back(fuzzy_set);
            }
        }
        fuzzy_vars[dim_i] = new FuzzyVariable(current_set, var_names[dim_i]);
    }
    knowledge->set_fuzzy_vars(fuzzy_vars);
    return knowledge;
}

HomoTriangleKnowledgeFactory* HomoTriangleKnowledgeFactory::clone() const {
    return new HomoTriangleKnowledgeFactory(*this);
}

std::vector<std::string> HomoTriangleKnowledgeFactory::get_fuzzy_set_names(int num_dims) {
    std::vector<std::string> var_names;
    for(int i=0; i<num_dims; i++)
    {
        var_names.push_back("x" + std::to_string(i));
    }
    return var_names;
}