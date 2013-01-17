#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "NodeDataCollectorI.h"


// Subsampling could live in here along with a max cap
class AllNodeDataCollector : public NodeDataCollectorI
{
public:
    AllNodeDataCollector();
    virtual ~AllNodeDataCollector();

    // Also copies/compacts weights, ys, etc
    virtual void Collect( BufferCollection& data,
                          const MatrixBufferInt& sampleIndices,
                          const MatrixBufferFloat& featureValues );

    // Includes feature values, weights, ys, etc
    virtual BufferCollection GetCollectedData();

    virtual int GetNumberOfCollectedSamples();
private:
    int mNumberOfCollectedSamples;
    BufferCollection mData;
};

class AllNodeDataCollectorFactory : public NodeDataCollectorFactoryI
{
public:
    virtual NodeDataCollectorI* Create() const;
};