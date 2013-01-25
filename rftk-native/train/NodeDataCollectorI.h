#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "MatrixBuffer.h"
#include "BufferCollection.h"


// Subsampling could live in here along with a max cap
class NodeDataCollectorI
{
public:
    virtual ~NodeDataCollectorI() {}

    // Also copies/compacts weights, ys, etc
    virtual void Collect( const BufferCollection& data,
                          const MatrixBufferInt& sampleIndices,
                          const MatrixBufferFloat& featureValues,
                          boost::mt19937& gen )
    {}

    // Includes feature values, weights, ys, etc
    virtual const BufferCollection& GetCollectedData()=0;

    virtual int GetNumberOfCollectedSamples() { return 0; }
};

class NodeDataCollectorFactoryI
{
public:
    virtual NodeDataCollectorFactoryI* Clone() const { return NULL; }
    virtual NodeDataCollectorI* Create() const { return NULL; }
};
