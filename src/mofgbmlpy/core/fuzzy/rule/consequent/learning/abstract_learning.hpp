#ifndef ABSTRACT_LEARNING_HPP
#define ABSTRACT_LEARNING_HPP

#include "../../../../data/dataset.hpp"
#include "../../antecedent/antecedent.hpp"
#include "../abstract_consequent.hpp"
#include <memory>


class AbstractLearning {
protected:
    std::shared_ptr<Dataset> train_ds;

public:
    AbstractLearning(std::shared_ptr<Dataset> training_dataset);

    virtual ~AbstractLearning() = default;

    virtual std::shared_ptr<AbstractConsequent> learning(
        Antecedent& antecedent,
        const Dataset& dataset,
        double reject_threshold = 0
    );
    virtual std::shared_ptr<AbstractConsequent> learning(
        Antecedent& antecedent,
        double reject_threshold = 0
    );
    virtual std::shared_ptr<Dataset> get_training_set() const;

    virtual AbstractLearning* clone() const = 0;
    virtual operator std::string() const = 0;
    virtual bool operator==(const AbstractLearning& other) const = 0;
    std::string to_string() const;
};


#endif  // ABSTRACT_LEARNING_HPP
