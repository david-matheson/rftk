#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

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
    virtual void Collect( const BufferCollection& data,
                          const Int32VectorBuffer& sampleIndices,
                          const Float32MatrixBuffer& featureValues,
                          boost::mt19937& gen);

    // Includes feature values, weights, ys, etc
    virtual const BufferCollection& GetCollectedData();

    virtual int GetNumberOfCollectedSamples();
private:
    BufferCollection mData;
};

class AllNodeDataCollectorFactory : public NodeDataCollectorFactoryI
{
public:
    virtual NodeDataCollectorFactoryI* Clone() const;
    virtual NodeDataCollectorI* Create() const;
};
