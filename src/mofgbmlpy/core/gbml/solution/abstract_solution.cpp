#include "abstract_solution.hpp"

AbstractSolution::AbstractSolution(int num_objectives, int num_constraints) {
    objectives.resize(num_objectives, 0.0);
    attributes = std::unordered_map<std::string, std::any>();
}

const std::vector<double>& AbstractSolution::getObjectives() const {
    return objectives;
}

void AbstractSolution::set_attribute(std::string& key, std::any& value) {
    attributes[key] = value;
}

std::any AbstractSolution::getAttribute(const std::string& key) const {
    return attributes.at(key);
}

bool AbstractSolution::hasAttribute(const std::string& id) const {
    return attributes.find(id) != attributes.end();
}

void AbstractSolution::set_objective(int index, double value) {
    objectives[index] = value;
}

double AbstractSolution::get_objective(int index) const {
    return objectives[index];
}

int AbstractSolution::get_num_objectives() const {
    return objectives.size();
}

int AbstractSolution::get_num_constraints() const {
    return 0; //TODO
}

std::unordered_map<std::string, std::any> AbstractSolution::get_attributes() {
    return attributes;
}

void AbstractSolution::clear_attributes() {
    attributes.clear();
}

std::string AbstractSolution::to_string() const {
    return static_cast<std::string>(*this);
}
