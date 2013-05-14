#pragma once

#include "BufferCollection.h"

class ActiveLeaf
{
public:
    ActiveLeaf()
    : mNodeData(NULL)
    , mIsInitialized(false)
    , mDatapointsSinceLastImpurityUpdate(0)
    {}

    ActiveLeaf(BufferCollection* nodeData)
    : mNodeData(nodeData)
    , mIsInitialized(false)
    , mDatapointsSinceLastImpurityUpdate(0)
    {}

    BufferCollection* mNodeData;
    bool mIsInitialized;
    int mDatapointsSinceLastImpurityUpdate;
};