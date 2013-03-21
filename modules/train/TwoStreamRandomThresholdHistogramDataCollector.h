#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <vector>
#include <set>

#include "buffers/MatrixBuffer.h"
#include "buffers/Tensor3Buffer.h"
#include "buffers/BufferCollection.h"
#include "RandomThresholdHistogramDataCollector.h"


// Subsampling could live in here along with a max cap
class TwoStreamRandomThresholdHistogramDataCollector : public RandomThresholdHistogramDataCollector
{
public:
    TwoStreamRandomThresholdHistogramDataCollector(int numberOfClasses,
                                                  int numberOfThresholds,
                                                  float probabilityOfNullStream,
                                                  float probabilityOfImpurityStream);
    virtual ~TwoStreamRandomThresholdHistogramDataCollector();

    // Also copies/compacts weights, ys, etc
    virtual void Collect( const BufferCollection& data,
                          const Int32VectorBuffer& sampleIndices,
                          const Float32MatrixBuffer& featureValues,
                          boost::mt19937& gen );

    // Includes feature values, weights, ys, etc
    virtual const BufferCollection& GetCollectedData();

    virtual int GetNumberOfCollectedSamples();
private:
    float mProbabilityOfImpurityStream;
};

class TwoStreamRandomThresholdHistogramDataCollectorFactory : public NodeDataCollectorFactoryI
{
public:
    TwoStreamRandomThresholdHistogramDataCollectorFactory(int numberOfClasses, int numberOfThresholds,
                                                          float probabilityOfNullStream, float probabilityOfImpurityStream);
    virtual ~TwoStreamRandomThresholdHistogramDataCollectorFactory();
    virtual NodeDataCollectorFactoryI* Clone() const;
    virtual NodeDataCollectorI* Create() const;

private:
    int mNumberOfClasses;
    int mNumberOfThresholds;
    float mProbabilityOfNullStream;
    float mProbabilityOfImpurityStream;
};
