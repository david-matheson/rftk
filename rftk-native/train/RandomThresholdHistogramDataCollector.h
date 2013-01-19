#pragma once

#include "MatrixBuffer.h"
#include "ImgBuffer.h"
#include "BufferCollection.h"
#include "NodeDataCollectorI.h"


// Subsampling could live in here along with a max cap
class RandomThresholdHistogramDataCollector : public NodeDataCollectorI
{
public:
    RandomThresholdHistogramDataCollector(int numberOfClasses, int numberOfThresholds, int numberOfSamplesToEstimateThresholds);
    virtual ~RandomThresholdHistogramDataCollector();

    // Also copies/compacts weights, ys, etc
    virtual void Collect( BufferCollection& data,
                          const MatrixBufferInt& sampleIndices,
                          const MatrixBufferFloat& featureValues );

    // Includes feature values, weights, ys, etc
    virtual BufferCollection GetCollectedData();

    virtual int GetNumberOfCollectedSamples();
private:
    int mNumberOfCollectedSamples;
    int mNumberOfClasses;
    int mNumberOfThresholds;
    int mNumberOfSamplesToEstimateThresholds;
    MatrixBufferFloat mFeaturesForThresholds;
    BufferCollection mData; // #features x #thresholds x #class

};

class RandomThresholdHistogramDataCollectorFactory : public NodeDataCollectorFactoryI
{
public:
    RandomThresholdHistogramDataCollectorFactory(int numberOfClasses, int numberOfThresholds, int numberOfSamplesToEstimateThresholds);
    virtual ~RandomThresholdHistogramDataCollectorFactory();
    virtual NodeDataCollectorI* Create() const;

private:
    int mNumberOfClasses;
    int mNumberOfThresholds;
    int mNumberOfSamplesToEstimateThresholds;
};
