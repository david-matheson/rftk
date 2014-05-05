#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "SplitBuffersI.h"
#include "FeatureEqualI.h"

template <class BufferTypes>
class SplitBuffersFeatureRange : public SplitBuffersI
{
public:
    SplitBuffersFeatureRange(const BufferId& pastFloatParamsBufferId,
                            const BufferId& pastIntParamsBufferId,
                            const BufferId& pastRangesBufferId,
                            const BufferId& initialRangeBufferId,
                            const FeatureEqualI<BufferTypes>* featureEqual);
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
    const FeatureEqualI<BufferTypes>* mFeatureEqual;
};

template <class BufferTypes>
SplitBuffersFeatureRange< BufferTypes>::SplitBuffersFeatureRange(const BufferId& pastFloatParamsBufferId,
                                                                      const BufferId& pastIntParamsBufferId,
                                                                      const BufferId& pastRangesBufferId,
                                                                      const BufferId& initialRangeBufferId,
                                                                      const FeatureEqualI<BufferTypes>* featureEqual )
: mPastFloatParamsBufferId(pastFloatParamsBufferId)
, mPastIntParamsBufferId(pastIntParamsBufferId)
, mPastRangesBufferId(pastRangesBufferId)
, mInitialRangeBufferId(initialRangeBufferId)
, mFeatureEqual(featureEqual->Clone())
{}

template <class BufferTypes>
SplitBuffersFeatureRange< BufferTypes>::SplitBuffersFeatureRange(const SplitBuffersFeatureRange& other )
: mPastFloatParamsBufferId(other.mPastFloatParamsBufferId)
, mPastIntParamsBufferId(other.mPastIntParamsBufferId)
, mPastRangesBufferId(other.mPastRangesBufferId)
, mInitialRangeBufferId(other.mInitialRangeBufferId)
, mFeatureEqual(other.mFeatureEqual->Clone())
{}

template <class BufferTypes>
SplitBuffersFeatureRange<BufferTypes>::~SplitBuffersFeatureRange()
{
    delete mFeatureEqual;
}

template <class BufferTypes>
void SplitBuffersFeatureRange<BufferTypes>::SplitBuffers(const SplitSelectorBuffers& splitSelectorBuffers,
                                                        int bestFeature,
                                                        int bestSplitpoint,
                                                        const BufferCollectionStack& readBuffers,
                                                        BufferCollection& leftBuffers, 
                                                        BufferCollection& rightBuffers) const
{
    const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams
           = readBuffers.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(splitSelectorBuffers.mFloatParamsBufferId);

    const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams
           = readBuffers.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(splitSelectorBuffers.mIntParamsBufferId); 

    const int NO_MATCH = -1;
    int matchIndex = NO_MATCH;

    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> pastFloatParams;
    MatrixBufferTemplate<typename BufferTypes::ParamsInteger> pastIntParams;
    MatrixBufferTemplate<typename BufferTypes::FeatureValue> pastRanges;

    if(!readBuffers.HasBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mPastFloatParamsBufferId))
    {
        // Create for the first time
        pastFloatParams = floatParams.SliceRow(bestFeature);
        pastIntParams = intParams.SliceRow(bestFeature);

        const VectorBufferTemplate<typename BufferTypes::SourceContinuous>& initialRanges =
              readBuffers.GetBuffer< VectorBufferTemplate<typename BufferTypes::SourceContinuous> >(mInitialRangeBufferId);
        pastRanges.Resize(1, initialRanges.GetN());
        pastRanges.SetRow(0, initialRanges);

        matchIndex = 0;
    }
    else
    {
        pastFloatParams = readBuffers.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mPastFloatParamsBufferId);
        pastIntParams = readBuffers.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mPastIntParamsBufferId);
        pastRanges = readBuffers.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(mPastRangesBufferId);

        ASSERT_ARG_DIM_1D(pastFloatParams.GetM(), pastIntParams.GetM())
        ASSERT_ARG_DIM_1D(pastFloatParams.GetM(), pastRanges.GetM())

        // Do a linear search to see if best params match any past params
        const typename BufferTypes::Index numberOfPastFeatures = pastFloatParams.GetM();
        for(typename BufferTypes::Index pastFeature=0; (pastFeature<numberOfPastFeatures) && (matchIndex == NO_MATCH); pastFeature++)
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

            const VectorBufferTemplate<typename BufferTypes::SourceContinuous>& initialRanges =
                  readBuffers.GetBuffer< VectorBufferTemplate<typename BufferTypes::SourceContinuous> >(mInitialRangeBufferId);
            pastRanges.AppendRow(initialRanges);

            matchIndex = pastRanges.GetM() - 1;

            ASSERT_ARG_DIM_1D(pastFloatParams.GetM(), pastIntParams.GetM())
            ASSERT_ARG_DIM_1D(pastFloatParams.GetM(), pastRanges.GetM())
        } 
    }

    const MatrixBufferTemplate<typename BufferTypes::FeatureValue>& splitpoints
          = readBuffers.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(splitSelectorBuffers.mSplitpointsBufferId);

    // Double check that the split point was the midpoint (the initial use for this is to implement Biau2012)
    const typename BufferTypes::FeatureValue midpoint = splitpoints.Get(bestFeature, bestSplitpoint);
    ASSERT(fabs(0.5*(pastRanges.Get(matchIndex,0)+pastRanges.Get(matchIndex,1)) - midpoint) < std::numeric_limits<typename BufferTypes::FeatureValue>::epsilon())

    // Update ranges for left and right children
    MatrixBufferTemplate<typename BufferTypes::FeatureValue> pastRangesLeft = pastRanges;
    pastRangesLeft.Set(matchIndex, 1, midpoint); //make midpoint lower bound for going left
    MatrixBufferTemplate<typename BufferTypes::FeatureValue> pastRangesRight = pastRanges;
    pastRangesRight.Set(matchIndex, 0, midpoint); //make midpoint upper bound for going right

    // Write out to left and right children buffers
    leftBuffers.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mPastFloatParamsBufferId, pastFloatParams );
    leftBuffers.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mPastIntParamsBufferId, pastIntParams );
    leftBuffers.AddBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(mPastRangesBufferId, pastRangesLeft );    

    rightBuffers.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mPastFloatParamsBufferId, pastFloatParams );
    rightBuffers.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mPastIntParamsBufferId, pastIntParams );
    rightBuffers.AddBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(mPastRangesBufferId, pastRangesRight ); 
}

template <class BufferTypes>
SplitBuffersI* SplitBuffersFeatureRange<BufferTypes>::Clone() const
{
    SplitBuffersFeatureRange<BufferTypes>* clone = new SplitBuffersFeatureRange<BufferTypes>(*this);
    return clone;
}