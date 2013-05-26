#pragma once

#include "BufferCollection.h"

template <class FloatType, class IntType>
class ActiveLeaf
{
public:
    ActiveLeaf(const int nodeIndex,
                        const IntType depth)
    : mNodeIndex(nodeIndex)
    , mDepth(depth)
    , mSplitBufferCollection()
    {}

    const int mNodeIndex;
    const IntType mDepth;
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