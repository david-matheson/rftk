#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <vector>
#include <set>

#include "MatrixBuffer.h"
#include "ImgBuffer.h"
#include "BufferCollection.h"
#include "NodeDataCollectorI.h"


// Subsampling could live in here along with a max cap
class RandomThresholdHistogramDataCollector : public NodeDataCollectorI
{
public:
    RandomThresholdHistogramDataCollector(  int numberOfClasses,
                                            int numberOfThresholds,
                                            float probabilityOfNullStream);
    virtual ~RandomThresholdHistogramDataCollector();

    // Also copies/compacts weights, ys, etc
    virtual void Collect( BufferCollection& data,
                          const MatrixBufferInt& sampleIndices,
                          const MatrixBufferFloat& featureValues,
                          boost::mt19937& gen );

    // Includes feature values, weights, ys, etc
    virtual BufferCollection GetCollectedData();

    virtual int GetNumberOfCollectedSamples();
private:
    int mNumberOfThresholdSamples;
    int mNumberOfCollectedSamples;
    int mNumberOfClasses;
    int mNumberOfThresholds;
    float mProbabilityOfNullStream;
    std::vector< std::set<float> > mCandidateThresholds;
    BufferCollection mData; // #features x #thresholds x #class
};

class RandomThresholdHistogramDataCollectorFactory : public NodeDataCollectorFactoryI
{
public:
    RandomThresholdHistogramDataCollectorFactory(int numberOfClasses, int numberOfThresholds, float probabilityOfNullStream);
    virtual ~RandomThresholdHistogramDataCollectorFactory();
    virtual NodeDataCollectorI* Create() const;

private:
    int mNumberOfClasses;
    int mNumberOfThresholds;
    float mProbabilityOfNullStream;
};
