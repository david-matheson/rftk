#pragma once

#include <limits>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>

#include "bootstrap.h"

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "FeatureExtractorStep.h"

// ----------------------------------------------------------------------------
//
// Select random split points at uniform from min/max feature value ranges
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class RandomUniformSplitpointsInRangeStep : public PipelineStepI
{
public:
    RandomUniformSplitpointsInRangeStep(const BufferId& featureValuesRangeMinMax, 
                                        int numberOfSplitpoints );


    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    const BufferId SplitpointsBufferId;
    const BufferId SplitpointsCountsBufferId;
private:
    const BufferId mFeatureValuesRangeMinMax;
    const int mNumberOfSplitpoints;

};

template <class BufferTypes>
RandomUniformSplitpointsInRangeStep<BufferTypes>::RandomUniformSplitpointsInRangeStep(const BufferId& featureValuesRangeMinMax, 
                                                                                    int numberOfSplitpoints )
: SplitpointsBufferId(GetBufferId("SplitpointsBufferId"))
, SplitpointsCountsBufferId(GetBufferId("SplitpointsCountsBufferId"))
, mFeatureValuesRangeMinMax(featureValuesRangeMinMax)
, mNumberOfSplitpoints(numberOfSplitpoints)
{}


template <class BufferTypes>
PipelineStepI* RandomUniformSplitpointsInRangeStep<BufferTypes>::Clone() const
{
    RandomUniformSplitpointsInRangeStep<BufferTypes>* clone = new RandomUniformSplitpointsInRangeStep<BufferTypes>(*this);
    return clone;
}


template <class BufferTypes>
void RandomUniformSplitpointsInRangeStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
                                        BufferCollection& writeCollection,
                                        boost::mt19937& gen,
                                        BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(extraInfo); //sampleIndicesWithOutReplacement does NOT currently use boost::mt19937
    UNUSED_PARAM(nodeIndex);

    const MatrixBufferTemplate<typename BufferTypes::FeatureValue>& featureValuesRangesMinMax =
          readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(mFeatureValuesRangeMinMax);

    MatrixBufferTemplate<typename BufferTypes::FeatureValue>& splitPoints =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(SplitpointsBufferId);

    VectorBufferTemplate<typename BufferTypes::Index>& splitPointsCounts =
            writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(SplitpointsCountsBufferId);

    const int numberOfFeatures = featureValuesRangesMinMax.GetM();
     
    splitPoints.Resize(numberOfFeatures, mNumberOfSplitpoints);
    splitPointsCounts.Resize(numberOfFeatures);

    boost::uniform_real<> uniform_splitpoint(0.0, 1.0);
    boost::variate_generator<boost::mt19937&,boost::uniform_real<> > var_uniform_splitpoint(gen, uniform_splitpoint);

    for(int featureIndex=0; featureIndex<numberOfFeatures; featureIndex++)
    {
        const typename BufferTypes::FeatureValue featureMin = featureValuesRangesMinMax.Get(featureIndex, 0);
        const typename BufferTypes::FeatureValue featureMax = featureValuesRangesMinMax.Get(featureIndex, 1);
       
        for(int splitPointIndex=0; splitPointIndex<mNumberOfSplitpoints; splitPointIndex++)
        {
            const float splitPoint = (featureMax - featureMin) * var_uniform_splitpoint() + featureMin;
            splitPoints.Set(featureIndex, splitPointIndex, splitPoint);
        }

        splitPointsCounts.Set(featureIndex, mNumberOfSplitpoints);
    }
}

