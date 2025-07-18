//
// Created by Robin on 12/07/2025.
//

#ifndef PITTSBURGH_SOLUTION_HPP
#define PITTSBURGH_SOLUTION_HPP
#include "../../fuzzy/classification/abstract_classification.hpp"


class MichiganSolutionBuilder;

class PittsburghSolution: public AbstractSolution {
private:
    std::shared_ptr<AbstractClassification> classification;
    std::shared_ptr<MichiganSolutionBuilder> michigan_solution_builder;
protected:
    std::vector<std::shared_ptr<MichiganSolution>> vars;
    double error_rate;
    std::vector<Pattern> errored_patterns;
public:
    PittsburghSolution(int num_vars, int num_objectives, int num_constraints,
                       std::shared_ptr<AbstractClassification> classification,
                       std::shared_ptr<MichiganSolutionBuilder> michigan_solution_builder = nullptr,
                       bool do_init_vars = true);
    PittsburghSolution(int num_objectives, int num_constraints,
                       std::shared_ptr<AbstractClassification> classification);
    PittsburghSolution(const PittsburghSolution& other);

    std::shared_ptr<MichiganSolutionBuilder> get_michigan_solution_builder();
    void learning(const Dataset& dataset);
    void learning();
    double get_average_rule_weight() const;
    void remove_vars(const std::vector<int>& indices);
    void clear_vars() override;
    std::vector<std::shared_ptr<MichiganSolution>> get_vars() const;
    std::shared_ptr<MichiganSolution> get_var(int index) const;
    void set_var(int index, std::shared_ptr<MichiganSolution> value);
    void set_vars(std::vector<std::shared_ptr<MichiganSolution>> new_vars);
    int get_num_vars() const override;
    bool are_rules_valid() const;
    std::shared_ptr<MichiganSolution> classify(const Pattern& pattern) const;
    int get_total_rule_length() const;
    void update_winners_and_errors(const Dataset& dataset);
    double calc_error_rate(const Dataset& dataset) const;
    double get_error_rate() const;
    const std::vector<Pattern>& get_errored_patterns() const;
    std::shared_ptr<AbstractClassification> get_classification() const;
    std::shared_ptr<AbstractClassLabel> predict(const Pattern& pattern) const;

    operator std::string() const override;
    bool operator==(const AbstractSolution& other) const override;
    PittsburghSolution* clone() const override;
};



#endif //PITTSBURGH_SOLUTION_HPP
