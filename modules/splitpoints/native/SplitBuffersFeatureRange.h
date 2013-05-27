#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "SplitBuffersI.h"
#include "FeatureEqualI.h"

template <class FloatType, class IntType>
class SplitBuffersFeatureRange : public SplitBuffersI
{
public:
    SplitBuffersFeatureRange(const BufferId& pastFloatParamsBufferId,
                            const BufferId& pastIntParamsBufferId,
                            const BufferId& pastRangesBufferId,
                            const BufferId& initialRangeBufferId,
                            const FeatureEqualI<FloatType, IntType>* featureEqual);
    SplitBuffersFeatureRange(const SplitBuffersFeatureRange& other);
    virtual ~SplitBuffersFeatureRange();

    virtual void SplitBuffers(const SplitSelectorBuffers& splitSelectorBuffers,
                              int bestFeature,
                              int bestSplitpoint,
                              const BufferCollectionStack& readBuffers,
                              BufferCollection& leftBuffers, 
                              BufferCollection& rightBuffers) const;

    virtual SplitBuffersI* Clone() const;

private:
    const BufferId mPastFloatParamsBufferId;
    const BufferId mPastIntParamsBufferId;
    const BufferId mPastRangesBufferId;
    const BufferId mInitialRangeBufferId;
    const FeatureEqualI<FloatType, IntType>* mFeatureEqual;
};

template <class FloatType, class IntType>
SplitBuffersFeatureRange< FloatType, IntType>::SplitBuffersFeatureRange(const BufferId& pastFloatParamsBufferId,
                                                                      const BufferId& pastIntParamsBufferId,
                                                                      const BufferId& pastRangesBufferId,
                                                                      const BufferId& initialRangeBufferId,
                                                                      const FeatureEqualI<FloatType, IntType>* featureEqual )
: mPastFloatParamsBufferId(pastFloatParamsBufferId)
, mPastIntParamsBufferId(pastIntParamsBufferId)
, mPastRangesBufferId(pastRangesBufferId)
, mInitialRangeBufferId(initialRangeBufferId)
, mFeatureEqual(featureEqual->Clone())
{}

template <class FloatType, class IntType>
SplitBuffersFeatureRange< FloatType, IntType>::SplitBuffersFeatureRange(const SplitBuffersFeatureRange& other )
: mPastFloatParamsBufferId(other.mPastFloatParamsBufferId)
, mPastIntParamsBufferId(other.mPastIntParamsBufferId)
, mPastRangesBufferId(other.mPastRangesBufferId)
, mInitialRangeBufferId(other.mInitialRangeBufferId)
, mFeatureEqual(other.mFeatureEqual->Clone())
{}

template <class FloatType, class IntType>
SplitBuffersFeatureRange<FloatType, IntType>::~SplitBuffersFeatureRange()
{
    delete mFeatureEqual;
}

template <class FloatType, class IntType>
void SplitBuffersFeatureRange<FloatType, IntType>::SplitBuffers(const SplitSelectorBuffers& splitSelectorBuffers,
                                                        int bestFeature,
                                                        int bestSplitpoint,
                                                        const BufferCollectionStack& readBuffers,
                                                        BufferCollection& leftBuffers, 
                                                        BufferCollection& rightBuffers) const
{
    const MatrixBufferTemplate<FloatType>& floatParams
           = readBuffers.GetBuffer< MatrixBufferTemplate<FloatType> >(splitSelectorBuffers.mFloatParamsBufferId);

    const MatrixBufferTemplate<IntType>& intParams
           = readBuffers.GetBuffer< MatrixBufferTemplate<IntType> >(splitSelectorBuffers.mIntParamsBufferId); 

    const int NO_MATCH = -1;
    int matchIndex = NO_MATCH;

    MatrixBufferTemplate<FloatType> pastFloatParams;
    MatrixBufferTemplate<IntType> pastIntParams;
    MatrixBufferTemplate<FloatType> pastRanges;

    if(!readBuffers.HasBuffer< MatrixBufferTemplate<FloatType> >(mPastFloatParamsBufferId))
    {
        // Create for the first time
        pastFloatParams = floatParams.SliceRow(bestFeature);
        pastIntParams = intParams.SliceRow(bestFeature);

        const VectorBufferTemplate<FloatType>& initialRanges =
              readBuffers.GetBuffer< VectorBufferTemplate<FloatType> >(mInitialRangeBufferId);
        pastRanges.Resize(1, initialRanges.GetN());
        pastRanges.SetRow(0, initialRanges);

        matchIndex = 0;
    }
    else
    {
        pastFloatParams = readBuffers.GetBuffer< MatrixBufferTemplate<FloatType> >(mPastFloatParamsBufferId);
        pastIntParams = readBuffers.GetBuffer< MatrixBufferTemplate<IntType> >(mPastIntParamsBufferId);
        pastRanges = readBuffers.GetBuffer< MatrixBufferTemplate<FloatType> >(mPastRangesBufferId);

        ASSERT_ARG_DIM_1D(pastFloatParams.GetM(), pastIntParams.GetM())
        ASSERT_ARG_DIM_1D(pastFloatParams.GetM(), pastRanges.GetM())

        // Do a linear search to see if best params match any past params
        const int numberOfPastFeatures = pastFloatParams.GetM();
        for(int pastFeature=0; (pastFeature<numberOfPastFeatures) && (matchIndex == NO_MATCH); pastFeature++)
        {
            const bool isEqual = mFeatureEqual->IsEqual(floatParams, intParams, bestFeature,
                                                        pastFloatParams, pastIntParams, pastFeature);
            if(isEqual)
            {
                matchIndex = pastFeature;
            }
        }

        // If the params are new, then add them
        if( matchIndex == NO_MATCH )
        {
            pastFloatParams.Append(floatParams.SliceRow(bestFeature));
            pastIntParams.Append(intParams.SliceRow(bestFeature));

            const VectorBufferTemplate<FloatType>& initialRanges =
                  readBuffers.GetBuffer< VectorBufferTemplate<FloatType> >(mInitialRangeBufferId);
            pastRanges.AppendRow(initialRanges);

            matchIndex = pastRanges.GetM() - 1;

            ASSERT_ARG_DIM_1D(pastFloatParams.GetM(), pastIntParams.GetM())
            ASSERT_ARG_DIM_1D(pastFloatParams.GetM(), pastRanges.GetM())
        } 
    }

    const MatrixBufferTemplate<FloatType>& splitpoints
          = readBuffers.GetBuffer< MatrixBufferTemplate<FloatType> >(splitSelectorBuffers.mSplitpointsBufferId);

    // Double check that the split point was the midpoint (the initial use for this is to implement Biau2012)
    const FloatType midpoint = splitpoints.Get(bestFeature, bestSplitpoint);
    ASSERT(fabs(0.5*(pastRanges.Get(matchIndex,0)+pastRanges.Get(matchIndex,1)) - midpoint) < std::numeric_limits<FloatType>::epsilon())

    // Update ranges for left and right children
    MatrixBufferTemplate<FloatType> pastRangesLeft = pastRanges;
    pastRangesLeft.Set(matchIndex, 1, midpoint); //make midpoint lower bound for going left
    MatrixBufferTemplate<FloatType> pastRangesRight = pastRanges;
    pastRangesRight.Set(matchIndex, 0, midpoint); //make midpoint upper bound for going right

    // Write out to left and right children buffers
    leftBuffers.AddBuffer< MatrixBufferTemplate<FloatType> >(mPastFloatParamsBufferId, pastFloatParams );
    leftBuffers.AddBuffer< MatrixBufferTemplate<IntType> >(mPastIntParamsBufferId, pastIntParams );
    leftBuffers.AddBuffer< MatrixBufferTemplate<FloatType> >(mPastRangesBufferId, pastRangesLeft );    

    rightBuffers.AddBuffer< MatrixBufferTemplate<FloatType> >(mPastFloatParamsBufferId, pastFloatParams );
    rightBuffers.AddBuffer< MatrixBufferTemplate<IntType> >(mPastIntParamsBufferId, pastIntParams );
    rightBuffers.AddBuffer< MatrixBufferTemplate<FloatType> >(mPastRangesBufferId, pastRangesRight ); 
}

template <class FloatType, class IntType>
SplitBuffersI* SplitBuffersFeatureRange<FloatType, IntType>::Clone() const
{
    SplitBuffersFeatureRange<FloatType, IntType>* clone = new SplitBuffersFeatureRange<FloatType, IntType>(*this);
    return clone;
}