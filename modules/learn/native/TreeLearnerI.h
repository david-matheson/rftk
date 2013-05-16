#pragma once

#include "BufferCollectionStack.h"
#include "Tree.h"

class TreeLearnerI 
{
public:
    virtual ~TreeLearnerI() {}
    virtual TreeLearnerI* Clone() const=0;
    virtual void Learn( const BufferCollection& data, Tree& tree, unsigned int seed) const=0;
};