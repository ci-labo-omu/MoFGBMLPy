#include "michigan_solution.hpp"

#include <sstream>
#include <utility>

MichiganSolution::MichiganSolution(std::shared_ptr<RandomGenerator> random_gen, int num_objectives, int num_constraints,
    std::shared_ptr<RuleBuilderCore> rule_builder, Pattern& pattern, bool do_init_vars): AbstractSolution(num_objectives, num_constraints), num_wins(0), fitness(0), rule_builder(std::move(rule_builder)), random_gen(std::move(random_gen)), are_scores_updated(false)
{
    if (do_init_vars) {
        create_rule(pattern);
        bool is_rejected = rule->is_rejected_class_label();
        int cnt = 0;
        std::shared_ptr<Dataset> training_data_set = rule_builder->get_training_dataset();

        while (is_rejected) {
            if (cnt > 1000) {
                throw std::runtime_error("Exceeded maximum number of trials to generate rule");
            }

            cnt ++;
            int pattern_idx = random_gen->randint(0, training_data_set->get_size());
            Pattern* pattern = training_data_set->get_pattern(pattern_idx);
            create_rule(*pattern);
            is_rejected = rule->is_rejected_class_label();
        }
    }
}

MichiganSolution::MichiganSolution(std::shared_ptr<RandomGenerator> random_gen, int num_objectives, int num_constraints,
    std::shared_ptr<RuleBuilderCore> rule_builder, bool do_init_vars): AbstractSolution(num_objectives, num_constraints), num_wins(0), fitness(0), rule_builder(std::move(rule_builder)), random_gen(std::move(random_gen)), are_scores_updated(false)
{
    if (do_init_vars) {
        int cnt = 0;
        bool is_rejected = true;
        while (is_rejected) {
            cnt++;
            create_rule();
            is_rejected = rule->is_rejected_class_label();
            if (cnt > 1000) {
                throw std::runtime_error("Exceeded maximum number of trials to generate rule");
            }
        }
    }
}

MichiganSolution::MichiganSolution(const MichiganSolution& other): AbstractSolution(other.get_num_objectives(), other.get_num_constraints()), num_wins(other.num_wins), fitness(other.fitness), rule_builder(other.rule_builder->clone()), random_gen(other.random_gen), are_scores_updated(other.are_scores_updated), rule(other.rule->clone()), vars(other.vars) {}

void MichiganSolution::set_scores_update_status(bool new_status)
{
    are_scores_updated = new_status;
}

void MichiganSolution::create_rule(const Pattern& pattern)
{
    set_vars(rule_builder->create_antecedent_indices(pattern)[0]);
    learning();
}

void MichiganSolution::create_rule()
{
    set_vars(rule_builder->create_antecedent_indices(1)[0]);
    learning();
}

void MichiganSolution::learning(const Dataset& dataset)
{
    if (rule == nullptr || rule->get_antecedent() == nullptr) {
        Antecedent* antecedent_object = rule_builder->create_antecedent_from_indices(vars);
        rule = std::shared_ptr<AbstractRule>(rule_builder->create(*antecedent_object));
    } else {
        std::shared_ptr<Antecedent> antecedent_object = rule->get_antecedent();
        antecedent_object->set_antecedent_indices(vars);
        rule->set_consequent(rule_builder->create_consequent(*antecedent_object, dataset));
    }
}

void MichiganSolution::learning()
{
    if (rule == nullptr || rule->get_antecedent() == nullptr) {
        Antecedent* antecedent_object = rule_builder->create_antecedent_from_indices(vars);
        rule = std::shared_ptr<AbstractRule>(rule_builder->create(*antecedent_object));
    } else {
        std::shared_ptr<Antecedent> antecedent_object = rule->get_antecedent();
        antecedent_object->set_antecedent_indices(vars);
        rule->set_consequent(rule_builder->create_consequent(*antecedent_object));
    }
}

double MichiganSolution::get_fitness_value(const std::vector<double>& in_vector) const
{
    return rule->get_fitness_value(in_vector);
}

int MichiganSolution::get_length() const
{
    return rule->get_length();
}

std::shared_ptr<AbstractClassLabel> MichiganSolution::get_class_label() const
{
    return rule->get_class_label();
}

std::shared_ptr<AbstractRuleWeight> MichiganSolution::get_rule_weight() const
{
    return rule->get_rule_weight();
}

std::shared_ptr<AbstractRule> MichiganSolution::get_rule() const
{
    return rule;
}

std::shared_ptr<RuleBuilderCore> MichiganSolution::get_rule_builder() const
{
    return rule_builder;
}

std::shared_ptr<AbstractConsequent> MichiganSolution::get_consequent() const
{
    return rule->get_consequent();
}

std::shared_ptr<Antecedent> MichiganSolution::get_antecedent() const
{
    return rule->get_antecedent();
}

std::vector<double> MichiganSolution::get_membership_values(const std::vector<double>& attribute_vector) const
{
    return rule->get_membership_values(attribute_vector);
}

double MichiganSolution::get_compatible_grade_value(const std::vector<double>& attribute_vector) const
{
    return rule->get_compatible_grade_value(attribute_vector);
}

void MichiganSolution::reset_num_wins()
{
    num_wins = 0;
    are_scores_updated = false;
}

void MichiganSolution::reset_fitness()
{
    num_wins = 0;
    are_scores_updated = false;
}

void MichiganSolution::inc_num_wins()
{
    num_wins++;
}

void MichiganSolution::inc_fitness()
{
    fitness++;
}

int MichiganSolution::get_num_wins() const
{
    if (!are_scores_updated) {
        throw std::runtime_error("Scores are not updated. Call learning() before accessing scores.");
    }
    return num_wins;
}

int MichiganSolution::get_fitness() const
{
    if (!are_scores_updated) {
        throw std::runtime_error("Scores are not updated. Call learning() before accessing scores.");
    }
    return fitness;
}

void MichiganSolution::clear_vars()
{
    vars = std::vector<int>();
    are_scores_updated = false;
}

std::vector<int> MichiganSolution::get_vars() const
{
    return vars;
}

int MichiganSolution::get_var(int index) const
{
    return vars[index];
}

void MichiganSolution::set_var(int index, int value)
{
    vars[index] = value;
    are_scores_updated = false;
}

void MichiganSolution::set_vars(const std::vector<int>& new_vars)
{
    vars = new_vars;
    are_scores_updated = false;
}

int MichiganSolution::get_num_vars() const
{
    return vars.size();
}

void MichiganSolution::set_antecedent_knowledge(Knowledge& knowledge)
{
    get_antecedent()->set_knowledge(&knowledge);
}

bool MichiganSolution::is_rejected()
{
    return rule->is_rejected_class_label();
}

MichiganSolution::operator std::string() const
{
    std::ostringstream oss;
    oss << "(MichiganSolution) Variables: [";

    for (int i = 0; i < get_num_vars(); ++i) {
        oss << vars[i] << " ";
    }

    oss << "], Rule weight: " << (rule ? std::string(*rule->get_rule_weight()) : "null")
        << ", Class label: " << (rule ? std::string(*rule->get_class_label()) : "null");

    oss << "], Objectives: [";
    for (int i = 0; i < get_num_objectives(); ++i) {
        oss << get_objective(i) << " ";
    }

    oss << "], Attributes: {Number of classifier patterns: " << fitness
        << ", Number of wins: " << num_wins;
    for (const auto& [key, val] : attributes) {
        oss << ", " << key << ": ";
        if (val.type() == typeid(std::string)) {
            oss << std::any_cast<std::string>(val);
        } else if (val.type() == typeid(int)) {
            oss << std::any_cast<int>(val);
        } else if (val.type() == typeid(double)) {
            oss << std::any_cast<double>(val);
        } else if (val.type() == typeid(float)) {
            oss << std::any_cast<float>(val);
        } else {
            oss << "<unknown type>";
        }
    }
    oss << "}";
    return oss.str();
}

bool MichiganSolution::operator==(const AbstractSolution& other) const
{
    MichiganSolution* other_michigan = dynamic_cast<MichiganSolution*>(&const_cast<AbstractSolution&>(other));
    if (other_michigan == nullptr) {
        return false;
    }
    if(vars != other_michigan->vars) {
        return false;
    }
    return true;
}

MichiganSolution* MichiganSolution::clone() const
{
    return new MichiganSolution(*this);
}
