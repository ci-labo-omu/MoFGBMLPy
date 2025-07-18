#include "abstract_learning.hpp"
#include <stdexcept>


AbstractLearning::AbstractLearning(std::shared_ptr<Dataset> training_dataset)
: train_ds(training_dataset)
{
    if (!training_dataset) {
        throw std::invalid_argument("Training dataset cannot be null");
    }
}

std::shared_ptr<Dataset> AbstractLearning::get_training_set() const {
    return train_ds;
}

std::string AbstractLearning::to_string() const {
    return static_cast<std::string>(*this);
}
