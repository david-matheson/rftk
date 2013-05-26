#pragma once

#include "UniqueBufferId.h"
#include "FeatureExtractorStep.h"

// ----------------------------------------------------------------------------
//
// All the buffers required to select, validate and write (to a tree) the best
// split point
//
// ----------------------------------------------------------------------------
class SplitSelectorBuffers
{
public:
    SplitSelectorBuffers();
    SplitSelectorBuffers(const BufferId& impurityBufferId,
                    const BufferId& splitpointsBufferId,
                    const BufferId& splitpointsCountsBufferId,
                    const BufferId& childCountsBufferId,
                    const BufferId& leftEstimatorParamsBufferId,
                    const BufferId& rightEstimatorParamsBufferId,
                    const BufferId& floatParamsBufferId,
                    const BufferId& intParamsBufferId,
                    const BufferId& featureValuesBufferId,
                    FeatureValueOrdering ordering);

    // BufferIds are non-const for vector assignment operator
    BufferId mImpurityBufferId;
    BufferId mSplitpointsBufferId;
    BufferId mSplitpointsCountsBufferId;
    BufferId mChildCountsBufferId;
    BufferId mLeftEstimatorParamsBufferId;
    BufferId mRightEstimatorParamsBufferId;
    BufferId mFloatParamsBufferId;
    BufferId mIntParamsBufferId;
    BufferId mFeatureValuesBufferId;
    FeatureValueOrdering mOrdering;
};

