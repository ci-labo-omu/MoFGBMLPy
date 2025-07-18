#include "triangular_mf.hpp"
#include <stdexcept>

TriangularMF::TriangularMF(float left, float center, float right) : AbstractMF({left, center, right}) {
    if (left > center) {
        throw std::invalid_argument("TriangularMF: left must be <= center");
    }
    if (center > right) {
        throw std::invalid_argument("TriangularMF: center must be <= right");
    }
}

float TriangularMF::get_value(float x) const {
    float left = params[0];
    float center = params[1];
    float right = params[2];

    if (x == center) return 1.0f; // For the case where left = center or center = right
    if (x <= left || x >= right) return 0.0f;
    if (x < center) return (x - left) / (center - left);
    return (right - x) / (right - center);
}

TriangularMF::operator std::string() const {
    return "<Triangular MF (" + std::to_string(params[0]) + ", " +
           std::to_string(params[1]) + ", " + std::to_string(params[2]) + ")>";
}

std::vector<float> TriangularMF::get_param_range(int index, float x_min, float x_max) const {
    if (x_min > params[0] || x_max < params[2]) {
        throw std::invalid_argument("TriangularMF: x_min and x_max must cover current parameters");
    }

    switch (index) {
        case 0: return { x_min, params[1] };
        case 1: return { params[0], params[2] };
        case 2: return { params[1], x_max };
        default:
            throw std::out_of_range("TriangularMF: invalid parameter index");
    }
}

std::vector<std::vector<float>> TriangularMF::get_plot_points(float x_min, float x_max) const {
    std::vector<std::vector<float>> points = {
        { x_min, 0.0f },
        { params[0], 0.0f },
        { params[1], 1.0f },
        { params[2], 0.0f },
        { x_max, 0.0f }
    };
    return points;
}

float TriangularMF::get_support(float x_min, float x_max) const {
    return 0.5f * (params[2] - params[0]);
}

TriangularMF* TriangularMF::clone() const {
    return new TriangularMF(params[0], params[1], params[2]);
}
