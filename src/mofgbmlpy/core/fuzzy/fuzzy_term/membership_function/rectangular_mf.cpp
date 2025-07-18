#include "rectangular_mf.hpp"
#include <stdexcept>

RectangularMF::RectangularMF(float left, float right) : AbstractMF({left, right}) {
    if (left > right) {
        throw std::invalid_argument("RectangularMF: left must be <= right");
    }
}

float RectangularMF::get_value(float x) const {
    float left = params[0];
    float right = params[1];
    if (x < left || x > right) return 0.0f;
    return 1.0f;
}

RectangularMF::operator std::string() const {
    return "<Rectangular MF>";
}

std::vector<float> RectangularMF::get_param_range(int index, float x_min, float x_max) const {
    if (x_min > params[0] || x_max < params[1]) {
        throw std::out_of_range("RectangularMF: x_min and x_max must be within the range of the membership function");
    }
    switch (index) {
        case 0: return { x_min, params[1] };
        case 1: return { params[0], x_max };
        default:
            throw std::out_of_range("RectangularMF: invalid parameter index");
    }
}

std::vector<std::vector<float>> RectangularMF::get_plot_points(float x_min, float x_max) const {
    std::vector<std::vector<float>> points = {
        { x_min, 0.0f },
        { params[0], 1.0f },
        { params[1], 1.0f },
        { x_max, 0.0f }
    };
    return points;
}

float RectangularMF::get_support(float x_min, float x_max) const {
    return params[1] - params[0];
}

RectangularMF* RectangularMF::clone() const {
    return new RectangularMF(params[0], params[1]);
}
