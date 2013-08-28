#pragma once

#include "UniqueBufferId.h"
#include "BufferCollectionStack.h"

class FeatureIndexerI
{
public:
    virtual int IndexFeature( const BufferCollectionStack& readCollection,
                              const int featureOffset ) const = 0;

    virtual FeatureIndexerI* CloneFeatureIndexerI() const = 0;

    virtual ~FeatureIndexerI() {};
};
