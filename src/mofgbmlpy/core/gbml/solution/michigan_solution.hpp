#ifndef MICHIGAN_SOLUTION_H
#define MICHIGAN_SOLUTION_H
#include "abstract_solution.hpp"
#include "../../fuzzy/rule/abstract_rule.hpp"
#include "../../fuzzy/rule/builder/rule_builder_core.hpp"
#include "../../utility/mersenne_twister_generator.hpp"


class MichiganSolution: public AbstractSolution {
private:
    int num_wins;
    int fitness;
protected:
    std::shared_ptr<AbstractRule> rule;
    std::shared_ptr<RuleBuilderCore> rule_builder;
    std::vector<int> vars;
    std::shared_ptr<RandomGenerator> random_gen;
    bool are_scores_updated;
public:
    MichiganSolution(std::shared_ptr<RandomGenerator> random_gen, int num_objectives, int num_constraints, std::shared_ptr<RuleBuilderCore> rule_builder, Pattern& pattern, bool do_init_vars = true);
    MichiganSolution(std::shared_ptr<RandomGenerator> random_gen, int num_objectives, int num_constraints, std::shared_ptr<RuleBuilderCore> rule_builder, bool do_init_vars = true);
    MichiganSolution(const MichiganSolution& other);

    void set_scores_update_status(bool new_status);
    void create_rule(const Pattern& pattern);
    void create_rule();
    void learning(const Dataset& dataset);
    void learning();
    double get_fitness_value(const std::vector<double>& in_vector) const;
    int get_length() const;
    std::shared_ptr<AbstractClassLabel> get_class_label() const;
    std::shared_ptr<AbstractRuleWeight> get_rule_weight() const;
    std::shared_ptr<AbstractRule> get_rule() const;
    std::shared_ptr<RuleBuilderCore> get_rule_builder() const;
    std::shared_ptr<AbstractConsequent> get_consequent() const;
    std::shared_ptr<Antecedent> get_antecedent() const;
    std::vector<double> get_membership_values(const std::vector<double>& attribute_vector) const;
    double get_compatible_grade_value(const std::vector<double>& attribute_vector) const;
    void reset_num_wins();
    void reset_fitness();
    void inc_num_wins();
    void inc_fitness();
    int get_num_wins() const;
    int get_fitness() const;
    void clear_vars() override;
    std::vector<int> get_vars() const;
    int get_var(int index) const;
    void set_var(int index, int value);
    void set_vars(const std::vector<int>& new_vars);
    int get_num_vars() const override;
    void set_antecedent_knowledge(Knowledge& knowledge);
    bool is_rejected();

    operator std::string() const override;
    bool operator==(const AbstractSolution& other) const override;
    MichiganSolution* clone() const override;
};



#endif //MICHIGAN_SOLUTION_H
