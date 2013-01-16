#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"


// Subsampling could live in here along with a max cap
class NodeDataCollectorI
{
public:
    virtual ~NodeDataCollectorI() {}

    // Also copies/compacts weights, ys, etc
    virtual void Collect( BufferCollection& data,
                          const MatrixBufferInt& sampleIndices,
                          const MatrixBufferFloat& featureValues )
    {
    }

    // Includes feature values, weights, ys, etc
    virtual BufferCollection GetCollectedData() { return BufferCollection(); }

    virtual int GetNumberOfCollectedSamples() { return 0; }
};

class NodeDataCollectorFactoryI
{
public:
    virtual NodeDataCollectorI* Create() const
    {
        return NULL;
    }
};
