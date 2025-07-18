#ifndef ABSTRACT_KNOWLEDGE_FACTORY_HPP
#define ABSTRACT_KNOWLEDGE_FACTORY_HPP

#include <memory>
#include "../knowledge.hpp"

class AbstractKnowledgeFactory {
public:
    virtual ~AbstractKnowledgeFactory() {}
    virtual std::shared_ptr<Knowledge> create() const = 0;
};

#endif
