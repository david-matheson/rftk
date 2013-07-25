#pragma once

#include "BufferTypes.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "FeatureEqualI.h"

// ----------------------------------------------------------------------------
//
// Biau 2012 halving split strategy
//
// Use the midpoint of the current range
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class RangeMidpointStep : public PipelineStepI
{
public:
    RangeMidpointStep(const BufferId& floatParamsBufferId,
                      const BufferId& intParamsBufferId,
                      const BufferId& initialRangeBufferId,
                      const FeatureEqualI<BufferTypes>* featureEqual );

    RangeMidpointStep( const RangeMidpointStep& other );

    ~RangeMidpointStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    const BufferId SplitpointsBufferId;
    const BufferId SplitpointsCountsBufferId;
    const BufferId PastFloatParamsBufferId;
    const BufferId PastIntParamsBufferId;
    const BufferId PastRangesBufferId;
private:
    const BufferId mFloatParamsBufferId;
    const BufferId mIntParamsBufferId;
    const BufferId mInitialRangeBufferId;
    const FeatureEqualI<BufferTypes>* mFeatureEqual;
};

template <class BufferTypes>
RangeMidpointStep< BufferTypes>::RangeMidpointStep(const BufferId& floatParamsBufferId,
                                                            const BufferId& intParamsBufferId,
                                                            const BufferId& initialRangeBufferId,
                                                            const FeatureEqualI<BufferTypes>* featureEqual )
: SplitpointsBufferId(GetBufferId("Splitpoints"))
, SplitpointsCountsBufferId(GetBufferId("SplitpointsCounts"))
, PastFloatParamsBufferId(GetBufferId("PastFloatParamsBufferId"))
, PastIntParamsBufferId(GetBufferId("PastIntParamsBufferId"))
, PastRangesBufferId(GetBufferId("PastRangesBufferId"))
, mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
, mInitialRangeBufferId(initialRangeBufferId)
, mFeatureEqual(featureEqual->Clone())
{}

template <class BufferTypes>
RangeMidpointStep< BufferTypes>::RangeMidpointStep(const RangeMidpointStep& other )
: SplitpointsBufferId(other.SplitpointsBufferId)
, SplitpointsCountsBufferId(other.SplitpointsCountsBufferId)
, PastFloatParamsBufferId(other.PastFloatParamsBufferId)
, PastIntParamsBufferId(other.PastIntParamsBufferId)
, PastRangesBufferId(other.PastRangesBufferId)
, mFloatParamsBufferId(other.mFloatParamsBufferId)
, mIntParamsBufferId(other.mIntParamsBufferId)
, mInitialRangeBufferId(other.mInitialRangeBufferId)
, mFeatureEqual(other.mFeatureEqual->Clone())
{}

template <class BufferTypes>
RangeMidpointStep< BufferTypes>::~RangeMidpointStep()
{
    delete mFeatureEqual;
}

template <class BufferTypes>
PipelineStepI* RangeMidpointStep< BufferTypes>::Clone() const
{
    RangeMidpointStep<BufferTypes>* clone = new RangeMidpointStep<BufferTypes>(*this);
    return clone;
}


template <class BufferTypes>
void RangeMidpointStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
                                        BufferCollection& writeCollection,
                                        boost::mt19937& gen) const
{
    UNUSED_PARAM(gen); //sampleIndicesWithOutReplacement does NOT currently use boost::mt19937

    const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams =
          readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFloatParamsBufferId);

    const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams =
          readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mIntParamsBufferId);

    const VectorBufferTemplate<typename BufferTypes::SourceContinuous>& initialRanges =
          readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::SourceContinuous> >(mInitialRangeBufferId);
    ASSERT_ARG_DIM_1D(initialRanges.GetN(), 2);

    MatrixBufferTemplate<typename BufferTypes::FeatureValue>& splitPoints =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(SplitpointsBufferId);

    VectorBufferTemplate<typename BufferTypes::Index>& splitPointsCounts =
            writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(SplitpointsCountsBufferId);

    const int numberOfFeatures = floatParams.GetM();
    splitPoints.Resize(numberOfFeatures, 1, 0.5*(initialRanges.Get(0)+initialRanges.Get(1))); // default to midpoint
    splitPointsCounts.Resize(numberOfFeatures);
    splitPointsCounts.SetAll(1);

    // If there are past parameter ranges then use midpoint of the current range
    int numberOfPastFeatures = 0;
    if( readCollection.HasBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(PastFloatParamsBufferId) )
    {
        const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& pastFloatParams =
              readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(PastFloatParamsBufferId);

        const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& pastIntParams =
              readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(PastIntParamsBufferId);

        const MatrixBufferTemplate<typename BufferTypes::FeatureValue>& pastRanges =
              readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(PastRangesBufferId);   
        
        numberOfPastFeatures = pastFloatParams.GetM();

        for(int feature=0; feature<numberOfFeatures; feature++)
        {
            for(int pastFeature=0; pastFeature<numberOfPastFeatures; pastFeature++)
            {   
                if(mFeatureEqual->IsEqual(pastFloatParams, pastIntParams, pastFeature, floatParams, intParams, feature))
                {
                    const typename BufferTypes::FeatureValue splitPoint = 0.5 * (pastRanges.Get(pastFeature,0) + pastRanges.Get(pastFeature, 1));
                    splitPoints.Set(feature, 0, splitPoint);
                  
                    // printf("Feature %d %d matches [%0.2f %0.2f] [%0.2f %0.2f]\n", pastFeature, feature,
                    //                                                               pastRanges.Get(pastFeature,0), splitPoint,
                    //                                                               splitPoint, pastRanges.Get(pastFeature,1));
                }
            }
        }       
    }
}