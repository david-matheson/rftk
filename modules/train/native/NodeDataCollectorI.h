#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <MatrixBuffer.h>
#include <BufferCollection.h>


// Subsampling could live in here along with a max cap
class NodeDataCollectorI
{
public:
    NodeDataCollectorI() {} //Needed by swig for pseudo abstract base classes
    virtual ~NodeDataCollectorI() {}; //Needed by swig for pseudo abstract base classes

    // Also copies/compacts weights, ys, etc
    virtual void Collect( const BufferCollection& data,
                          const Int32VectorBuffer& sampleIndices,
                          const Float32MatrixBuffer& featureValues,
                          boost::mt19937& gen )=0;

    // Includes feature values, weights, ys, etc
    virtual const BufferCollection& GetCollectedData()=0;

    virtual int GetNumberOfCollectedSamples()=0;
};

class NodeDataCollectorFactoryI
{
public:
    virtual ~NodeDataCollectorFactoryI() {};
    virtual NodeDataCollectorFactoryI* Clone() const=0;
    virtual NodeDataCollectorI* Create() const=0;
};
