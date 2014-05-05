#pragma once

#include "UniqueBufferId.h"
#include "BufferCollectionStack.h"

class FeatureInfoLoggerI
{
public:
    virtual void LogFeatureInfo( const BufferCollectionStack& readCollection, int depth,
		                        const int featureOffset, const double featureImpurity, const bool isSelectedFeature, 
		                        BufferCollection& extraInfo) const = 0;

    virtual FeatureInfoLoggerI* CloneFeatureInfoLoggerI() const = 0;

    virtual ~FeatureInfoLoggerI() {};
};
