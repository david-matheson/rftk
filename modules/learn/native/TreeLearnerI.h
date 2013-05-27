#pragma once

#include "BufferCollectionStack.h"
#include "Tree.h"

class TreeLearnerI 
{
public:
    virtual ~TreeLearnerI() {}
    virtual TreeLearnerI* Clone() const=0;
    virtual void Learn( BufferCollectionStack stack, Tree& tree, unsigned int seed) const=0;
};