#pragma once

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
template <class FloatType, class IntType>
class RangeMidpointStep : public PipelineStepI
{
public:
    RangeMidpointStep(const BufferId& floatParamsBufferId,
                      const BufferId& intParamsBufferId,
                      const BufferId& initialRangeBufferId,
                      const FeatureEqualI<FloatType, IntType>* featureEqual );

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
    const FeatureEqualI<FloatType, IntType>* mFeatureEqual;
};

template <class FloatType, class IntType>
RangeMidpointStep< FloatType, IntType>::RangeMidpointStep(const BufferId& floatParamsBufferId,
                                                            const BufferId& intParamsBufferId,
                                                            const BufferId& initialRangeBufferId,
                                                            const FeatureEqualI<FloatType, IntType>* featureEqual )
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

template <class FloatType, class IntType>
RangeMidpointStep< FloatType, IntType>::RangeMidpointStep(const RangeMidpointStep& other )
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

template <class FloatType, class IntType>
RangeMidpointStep< FloatType, IntType>::~RangeMidpointStep()
{
    delete mFeatureEqual;
}

template <class FloatType, class IntType>
PipelineStepI* RangeMidpointStep< FloatType, IntType>::Clone() const
{
    RangeMidpointStep<FloatType, IntType>* clone = new RangeMidpointStep<FloatType, IntType>(*this);
    return clone;
}


template <class FloatType, class IntType>
void RangeMidpointStep<FloatType, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                        BufferCollection& writeCollection,
                                        boost::mt19937& gen) const
{
    UNUSED_PARAM(gen); //sampleIndicesWithOutReplacement does NOT currently use boost::mt19937

    const MatrixBufferTemplate<FloatType>& floatParams =
          readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(mFloatParamsBufferId);

    const MatrixBufferTemplate<IntType>& intParams =
          readCollection.GetBuffer< MatrixBufferTemplate<IntType> >(mIntParamsBufferId);

    const VectorBufferTemplate<FloatType>& initialRanges =
          readCollection.GetBuffer< VectorBufferTemplate<FloatType> >(mInitialRangeBufferId);
    ASSERT_ARG_DIM_1D(initialRanges.GetN(), 2);

    MatrixBufferTemplate<FloatType>& splitPoints =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<FloatType> >(SplitpointsBufferId);

    VectorBufferTemplate<IntType>& splitPointsCounts =
            writeCollection.GetOrAddBuffer< VectorBufferTemplate<IntType> >(SplitpointsCountsBufferId);

    const int numberOfFeatures = floatParams.GetM();
    splitPoints.Resize(numberOfFeatures, 1, 0.5*(initialRanges.Get(0)+initialRanges.Get(1))); // default to midpoint
    splitPointsCounts.Resize(numberOfFeatures);
    splitPointsCounts.SetAll(1);

    // If there are past parameter ranges then use midpoint of the current range
    int numberOfPastFeatures = 0;
    if( readCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(PastFloatParamsBufferId) )
    {
        const MatrixBufferTemplate<FloatType>& pastFloatParams =
              readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(PastFloatParamsBufferId);

        const MatrixBufferTemplate<IntType>& pastIntParams =
              readCollection.GetBuffer< MatrixBufferTemplate<IntType> >(PastIntParamsBufferId);

        const MatrixBufferTemplate<FloatType>& pastRanges =
              readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(PastRangesBufferId);   
        
        numberOfPastFeatures = pastFloatParams.GetM();

        for(int feature=0; feature<numberOfFeatures; feature++)
        {
            for(int pastFeature=0; pastFeature<numberOfPastFeatures; pastFeature++)
            {   
                if(mFeatureEqual->IsEqual(pastFloatParams, pastIntParams, pastFeature, floatParams, intParams, feature))
                {
                    const FloatType splitPoint = 0.5 * (pastRanges.Get(pastFeature,0) + pastRanges.Get(pastFeature, 1));
                    splitPoints.Set(feature, 0, splitPoint);
                  
                    // printf("Feature %d %d matches [%0.2f %0.2f] [%0.2f %0.2f]\n", pastFeature, feature,
                    //                                                               pastRanges.Get(pastFeature,0), splitPoint,
                    //                                                               splitPoint, pastRanges.Get(pastFeature,1));
                }
            }
        }       
    }
}