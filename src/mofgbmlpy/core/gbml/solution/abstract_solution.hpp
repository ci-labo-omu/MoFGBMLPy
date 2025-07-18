#ifndef ABSTRACT_SOLUTION_H
#define ABSTRACT_SOLUTION_H

#include <unordered_map>
#include <any>
#include <string>
#include <vector>
#include <stdexcept>

class AbstractSolution {
protected:
    std::vector<double> objectives;
    // std::vector<double> constraints;
    std::unordered_map<std::string, std::any> attributes;

public:
    AbstractSolution(int num_objectives, int num_constraints=0);
    virtual ~AbstractSolution() = default;

    const std::vector<double>& getObjectives() const;
    void set_attribute(std::string& key, std::any& value);
    std::any getAttribute(const std::string& id) const;
    bool hasAttribute(const std::string& id) const;
    void set_objective(int index, double value);
    double get_objective(int index) const;
    virtual int get_num_vars() const = 0;
    virtual void clear_vars() = 0;
    int get_num_objectives() const;
    int get_num_constraints() const;
    std::unordered_map<std::string, std::any> get_attributes();
    void clear_attributes();

    virtual operator std::string() const = 0;
    std::string to_string() const;
    virtual bool operator==(const AbstractSolution& other) const = 0;
    virtual AbstractSolution* clone() const = 0;
};



#endif //ABSTRACT_SOLUTION_H
