#pragma once

#include "BufferCollection.h"

class ActiveLeaf
{
public:
    ActiveLeaf(const int nodeIndex,
                const int depth)
    : mNodeIndex(nodeIndex)
    , mDepth(depth)
    , mSplitBufferCollection()
    {}

    const int mNodeIndex;
    const int mDepth;
    BufferCollection mSplitBufferCollection;
};


class ActiveOnlineLeaf
{
public:
    ActiveOnlineLeaf()
    : mNodeData(NULL)
    , mIsInitialized(false)
    , mDatapointsSinceLastImpurityUpdate(0)
    {}

    ActiveOnlineLeaf(BufferCollection* nodeData)
    : mNodeData(nodeData)
    , mIsInitialized(false)
    , mDatapointsSinceLastImpurityUpdate(0)
    {}

    BufferCollection* mNodeData;
    bool mIsInitialized;
    int mDatapointsSinceLastImpurityUpdate;
};