#pragma once

#include <limits>
#include "bootstrap.h"

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "FeatureExtractorStep.h"
#include "AssignStreamStep.h"

// ----------------------------------------------------------------------------
//
// Select random split points from feature values
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class RandomSplitpointsStep : public PipelineStepI
{
public:
    RandomSplitpointsStep(const BufferId& featureValuesBufferId, 
                          int maxSplitpointsPerFeature, 
                          FeatureValueOrdering featureValueOrdering );

    RandomSplitpointsStep(const BufferId& featureValuesBufferId, 
                          int maxSplitpointsPerFeature, 
                          FeatureValueOrdering featureValueOrdering,
                          const BufferId& streamTypeBufferId );

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    const BufferId SplitpointsBufferId;
    const BufferId SplitpointsCountsBufferId;
private:
    bool IsFull(const VectorBufferTemplate<typename BufferTypes::Index>& splitPointsCounts) const;

    void AddSplitpoint(MatrixBufferTemplate<typename BufferTypes::FeatureValue>& splitPoints,
                        VectorBufferTemplate<typename BufferTypes::Index>& splitPointsCounts,
                        const int featureIndex,
                        const typename BufferTypes::FeatureValue featureValue) const;

    const BufferId mFeatureValuesBufferId;
    const int mMaxSplitpointPerFeature;
    const FeatureValueOrdering mFeatureValueOrdering;
    const BufferId mStreamTypeBufferId;
};

template <class BufferTypes>
RandomSplitpointsStep<BufferTypes>::RandomSplitpointsStep(const BufferId& featureValuesBufferId, 
                                                            int maxSplitpointsPerFeature, 
                                                            FeatureValueOrdering featureValueOrdering )
: SplitpointsBufferId(GetBufferId("Splitpoints"))
, SplitpointsCountsBufferId(GetBufferId("SplitpointsCounts"))
, mFeatureValuesBufferId(featureValuesBufferId)
, mMaxSplitpointPerFeature(maxSplitpointsPerFeature)
, mFeatureValueOrdering(featureValueOrdering)
, mStreamTypeBufferId("NoStreamType")
{}

template <class BufferTypes>
RandomSplitpointsStep<BufferTypes>::RandomSplitpointsStep(const BufferId& featureValuesBufferId, 
                                                            int maxSplitpointsPerFeature, 
                                                            FeatureValueOrdering featureValueOrdering,
                                                            const BufferId& streamTypeBufferId )
: SplitpointsBufferId(GetBufferId("Splitpoints"))
, SplitpointsCountsBufferId(GetBufferId("SplitpointsCounts"))
, mFeatureValuesBufferId(featureValuesBufferId)
, mMaxSplitpointPerFeature(maxSplitpointsPerFeature)
, mFeatureValueOrdering(featureValueOrdering)
, mStreamTypeBufferId(streamTypeBufferId)
{}


template <class BufferTypes>
PipelineStepI* RandomSplitpointsStep<BufferTypes>::Clone() const
{
    RandomSplitpointsStep<BufferTypes>* clone = new RandomSplitpointsStep<BufferTypes>(*this);
    return clone;
}


template <class BufferTypes>
void RandomSplitpointsStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
                                        BufferCollection& writeCollection,
                                        boost::mt19937& gen) const
{
    UNUSED_PARAM(gen); //sampleIndicesWithOutReplacement does NOT currently use boost::mt19937

    const MatrixBufferTemplate<typename BufferTypes::FeatureValue>& featureValues =
          readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(mFeatureValuesBufferId);

    MatrixBufferTemplate<typename BufferTypes::FeatureValue>& splitPoints =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(SplitpointsBufferId);

    VectorBufferTemplate<typename BufferTypes::Index>& splitPointsCounts =
            writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(SplitpointsCountsBufferId);

    VectorBufferTemplate<typename BufferTypes::ParamsInteger> const* streamType = NULL;
    if(readCollection.HasBuffer< VectorBufferTemplate<typename BufferTypes::ParamsInteger> >(mStreamTypeBufferId))
    {
        streamType = readCollection.GetBufferPtr< VectorBufferTemplate<typename BufferTypes::ParamsInteger> >(mStreamTypeBufferId);
    }

    const int numberOfFeatures =  mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? featureValues.GetM() : featureValues.GetN();
    const int numberOfSamples =  mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? featureValues.GetN() : featureValues.GetM();
   
    splitPoints.Resize(numberOfFeatures, mMaxSplitpointPerFeature);
    splitPointsCounts.Resize(numberOfFeatures);

    if(IsFull(splitPointsCounts)) return;

    std::vector<int> randomOrder(numberOfSamples);
    sampleIndicesWithOutReplacement(&randomOrder[0], randomOrder.size(), randomOrder.size());

    for(int j=0; j<numberOfSamples && !IsFull(splitPointsCounts); j++)
    {
        const int i = randomOrder.at(j);
        if(streamType == NULL || streamType->Get(i) == STREAM_STRUCTURE)
        {
            for(int f=0; f<numberOfFeatures; f++)
            {
                const int r = (mFeatureValueOrdering == FEATURES_BY_DATAPOINTS) ? f : i;
                const int c = (mFeatureValueOrdering == FEATURES_BY_DATAPOINTS) ? i : f;
                const float featureValue = featureValues.Get(r,c);
                AddSplitpoint(splitPoints, splitPointsCounts, f, featureValue);
            }
        }
    }
}

template <class BufferTypes>
bool RandomSplitpointsStep<BufferTypes>::IsFull(const VectorBufferTemplate<typename BufferTypes::Index>& splitPointsCounts) const
{
    bool isFull = true;
    for(int f=0; f<splitPointsCounts.GetN() && isFull; f++)
    {
        isFull = isFull && (splitPointsCounts.Get(f) >= mMaxSplitpointPerFeature);
    }
    return isFull;
}

template <class BufferTypes>
void RandomSplitpointsStep<BufferTypes>::AddSplitpoint(MatrixBufferTemplate<typename BufferTypes::FeatureValue>& splitPoints,
                                                              VectorBufferTemplate<typename BufferTypes::Index>& splitPointsCounts,
                                                              const int featureIndex,
                                                              const typename BufferTypes::FeatureValue featureValue) const
{
    const int splitPointCount = splitPointsCounts.Get(featureIndex);
    if( splitPointCount < mMaxSplitpointPerFeature )
    {
        for(int i=0; i<splitPointCount; i++)
        {
            if(fabs(featureValue - splitPoints.Get(featureIndex, i)) < std::numeric_limits<float>::epsilon())
            {
                return;
            }
        }

        splitPoints.Set(featureIndex, splitPointCount, featureValue);
        splitPointsCounts.Incr(featureIndex, 1);
    }
}