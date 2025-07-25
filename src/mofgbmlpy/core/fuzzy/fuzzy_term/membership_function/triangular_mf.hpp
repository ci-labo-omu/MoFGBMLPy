#pragma once
#include <vector>
#include "abstract_mf.hpp"
#include <string>

class TriangularMF : public AbstractMF {
public:
    TriangularMF(float left = 0.0f, float center = 0.5f, float right = 1.0f);

    float get_value(float x) const override;
    std::vector<float> get_param_range(int index, float x_min = 0, float x_max = 1) const override;
    std::vector<std::vector<float>> get_plot_points(float x_min = 0, float x_max = 1) const override;
    float get_support(float x_min = 0, float x_max = 0) const override;

    TriangularMF* clone() const override;
    operator std::string() const override;
};
