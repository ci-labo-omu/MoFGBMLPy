//
// Created by Robin on 12/07/2025.
//

#include "pittsburgh_solution.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#include "builder/michigan_solution_builder.hpp"
#include "../../fuzzy/rule/consequent/ruleWeight/rule_weight_multi.hpp"

PittsburghSolution::PittsburghSolution(int num_vars, int num_objectives, int num_constraints,
                                       std::shared_ptr<AbstractClassification> classification,
                                       std::shared_ptr<MichiganSolutionBuilder> michigan_solution_builder, bool do_init_vars) : AbstractSolution(num_objectives, num_constraints), michigan_solution_builder(std::move(michigan_solution_builder)), classification(std::move(classification)), error_rate(-1), errored_patterns(std::vector<Pattern>())
{
    if (do_init_vars) {
        vars = michigan_solution_builder->create(num_vars);
    }
}

PittsburghSolution::PittsburghSolution(int num_objectives, int num_constraints,
    std::shared_ptr<AbstractClassification> classification)
    : AbstractSolution(num_objectives, num_constraints), classification(std::move(classification)), error_rate(-1), errored_patterns(std::vector<Pattern>()) {}

PittsburghSolution::PittsburghSolution(const PittsburghSolution& other) : AbstractSolution(other.get_num_objectives(), other.get_num_constraints()),
    classification(other.classification->clone()), michigan_solution_builder(other.michigan_solution_builder->clone()),
    error_rate(other.error_rate), errored_patterns(other.errored_patterns)
{
    vars = std::vector<std::shared_ptr<MichiganSolution>>();
    for (const auto& var : other.vars) {
        vars.push_back(std::shared_ptr<MichiganSolution>(var->clone()));
    }
}

std::shared_ptr<MichiganSolutionBuilder> PittsburghSolution::get_michigan_solution_builder()
{
    return michigan_solution_builder;
}

void PittsburghSolution::learning(const Dataset& dataset)
{
    for (auto& var : vars) {
        var->learning(dataset);
    }
}

void PittsburghSolution::learning()
{
    for (auto& var : vars) {
        var->learning();
    }
}

double PittsburghSolution::get_average_rule_weight() const
{
    if (vars.empty()) {
        throw std::runtime_error("No variables in PittsburghSolution to calculate average rule weight.");
    }

    double total_rule_weight = 0.0;

    for (const auto& var : vars) {
        std::shared_ptr<AbstractRuleWeight> rule_weight = var->get_rule_weight();
        RuleWeightMulti* rule_weight_multi = dynamic_cast<RuleWeightMulti*>(rule_weight.get());

        if (rule_weight_multi != nullptr) {
            total_rule_weight += rule_weight_multi->get_mean();
        } else {
            total_rule_weight += std::get<double>(rule_weight->get_value());
        }
    }
    return total_rule_weight / vars.size();
}

void PittsburghSolution::remove_vars(const std::vector<int>& indices)
{
    std::vector<std::shared_ptr<MichiganSolution>> new_vars;
    new_vars.reserve(vars.size() - indices.size());
    for (int i = 0; i < vars.size(); ++i) {
        if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
            new_vars.push_back(vars[i]);
        }
    }
}

void PittsburghSolution::clear_vars()
{
    vars.clear();
}

std::vector<std::shared_ptr<MichiganSolution>> PittsburghSolution::get_vars() const
{
    return vars;
}

std::shared_ptr<MichiganSolution> PittsburghSolution::get_var(int index) const
{
    return vars.at(index);
}

void PittsburghSolution::set_var(int index, const std::shared_ptr<MichiganSolution> value)
{
    vars[index] = value;
}

void PittsburghSolution::set_vars(const std::vector<std::shared_ptr<MichiganSolution>> new_vars)
{
    vars = new_vars;
}

int PittsburghSolution::get_num_vars() const
{
    return vars.size();
}

bool PittsburghSolution::are_rules_valid() const
{
    if (vars.empty()) {
        return false;
    }

    for (const auto& var : vars) {
        if (var->get_rule()->is_rejected_class_label()) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<MichiganSolution> PittsburghSolution::classify(const Pattern& pattern) const
{
    return classification->classify(vars, pattern);
}

int PittsburghSolution::get_total_rule_length() const
{
    int length = 0;
    for (const auto& var : vars) {
        length += var->get_length();
    }
    return length;
}

void PittsburghSolution::update_winners_and_errors(const Dataset& dataset)
{
    int dataset_size = dataset.get_size();
    std::vector<Pattern*> patterns = dataset.get_patterns();
    std::vector<int> errored_patterns_indices;
    int num_errors = 0;

    for (auto& var : vars) {
        var->reset_num_wins();
        var->reset_fitness();
    }

    for (int i = 0; i < dataset_size; ++i) {
        Pattern* pattern = patterns[i];
        std::shared_ptr<MichiganSolution> winner = classify(*pattern);

        if (winner == nullptr) {
            num_errors++;
            errored_patterns_indices.push_back(i);
            continue;
        }

        winner->inc_num_wins();

        if (pattern->get_target_class() != winner->get_class_label()) {
            num_errors++;
            errored_patterns_indices.push_back(i);
        } else {
            winner->inc_fitness();
        }
    }

    errored_patterns.clear();
    for (int& index : errored_patterns_indices) {
        errored_patterns.push_back(*patterns[index]);
    }

    for (auto& var : vars) {
        var->set_scores_update_status(true);
    }
    error_rate = static_cast<double>(num_errors) / dataset_size;
}

double PittsburghSolution::calc_error_rate(const Dataset& dataset) const
{
    int dataset_size = dataset.get_size();
    std::vector<Pattern*> patterns = dataset.get_patterns();
    int num_errors = 0;

    for (int i = 0; i < dataset_size; ++i) {
        Pattern* pattern = patterns[i];
        std::shared_ptr<MichiganSolution> winner = classify(*pattern);

        if (winner == nullptr || pattern->get_target_class() != winner->get_class_label()) {
            num_errors++;
        }
    }

    return static_cast<double>(num_errors) / dataset_size;
}

double PittsburghSolution::get_error_rate() const
{
    if (error_rate < 0) {
        throw std::runtime_error("Error rate was not initialized. Please call update_winners_and_errors first");
    }
    return error_rate;
}

const std::vector<Pattern>& PittsburghSolution::get_errored_patterns() const
{
    return errored_patterns;
}

std::shared_ptr<AbstractClassification> PittsburghSolution::get_classification() const
{
    return classification;
}

std::shared_ptr<AbstractClassLabel> PittsburghSolution::predict(const Pattern& pattern) const
{
    std::shared_ptr<MichiganSolution> winner = classify(pattern);
    if (winner == nullptr) {
        return nullptr;
    }
    return winner->get_class_label();
}

PittsburghSolution::operator std::string() const
{
    std::ostringstream oss;
    oss << "(PittsburghSolution) Variables: [";
    for (int i = 0; i < get_num_vars(); ++i) {
        oss << std::string(*vars[i]) << " ";
    }
    oss << "], Objectives: [";
    for (int i = 0; i < get_num_objectives(); ++i) {
        oss << get_objective(i) << " ";
    }
    oss << "], Attributes: {";
    for (const auto& [key, val] : attributes) {
        oss << key << ": ";
        if (val.type() == typeid(std::string)) {
            oss << std::any_cast<std::string>(val);
        } else if (val.type() == typeid(int)) {
            oss << std::any_cast<int>(val);
        } else if (val.type() == typeid(double)) {
            oss << std::any_cast<double>(val);
        } else {
            oss << "unknown type";
        }
        oss << ", ";
    }
    oss << "}";
    return oss.str();
}

bool PittsburghSolution::operator==(const AbstractSolution& other) const
{
    const PittsburghSolution* other_pittsburgh = dynamic_cast<const PittsburghSolution*>(&other);
    if (other_pittsburgh == nullptr) {
        return false;
    }

    return vars == other_pittsburgh->vars;
}

PittsburghSolution* PittsburghSolution::clone() const
{
    return new PittsburghSolution(*this);
}
