#pragma once

#include "BufferCollection.h"
#include "SplitSelectorInfo.h"

template <class FloatType, class IntType>
class SplitSelectorI
{
  public:
    virtual ~SplitSelectorI() {}
    virtual SplitSelectorInfo<FloatType, IntType> ProcessSplits(const BufferCollectionStack& bufferCollectionStack, int depth) const=0;
    virtual SplitSelectorI<FloatType, IntType>* Clone() const=0;
};