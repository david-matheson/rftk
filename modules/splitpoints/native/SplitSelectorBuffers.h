#pragma once

#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// All the buffers required to select, validate and write (to a tree) the best
// split point 
//
// ----------------------------------------------------------------------------
class SplitSelectorBuffers 
{
public:
    SplitSelectorBuffers(const UniqueBufferId::BufferId& impurityBufferId,
                    const UniqueBufferId::BufferId& splitpointsBufferId,
                    const UniqueBufferId::BufferId& splitpointsCountsBufferId,
                    const UniqueBufferId::BufferId& childCountsBufferId,
                    const UniqueBufferId::BufferId& leftEstimatorParamsBufferId,
                    const UniqueBufferId::BufferId& rightEstimatorParamsBufferId,
                    const UniqueBufferId::BufferId& floatParamsBufferId,
                    const UniqueBufferId::BufferId& intParamsBufferId);
 
    // BufferIds are non-const for vector assignment operator
    UniqueBufferId::BufferId mImpurityBufferId;
    UniqueBufferId::BufferId mSplitpointsBufferId;
    UniqueBufferId::BufferId mSplitpointsCountsBufferId;
    UniqueBufferId::BufferId mChildCountsBufferId;
    UniqueBufferId::BufferId mLeftEstimatorParamsBufferId;
    UniqueBufferId::BufferId mRightEstimatorParamsBufferId;
    UniqueBufferId::BufferId mFloatParamsBufferId;
    UniqueBufferId::BufferId mIntParamsBufferId;
};


SplitSelectorBuffers::SplitSelectorBuffers(const UniqueBufferId::BufferId& impurityBufferId,
                                const UniqueBufferId::BufferId& splitpointsBufferId,
                                const UniqueBufferId::BufferId& splitpointsCountsBufferId,
                                const UniqueBufferId::BufferId& childCountsBufferId,
                                const UniqueBufferId::BufferId& leftEstimatorParamsBufferId,
                                const UniqueBufferId::BufferId& rightEstimatorParamsBufferId,
                                const UniqueBufferId::BufferId& floatParamsBufferId,
                                const UniqueBufferId::BufferId& intParamsBufferId)
: mImpurityBufferId(impurityBufferId)
, mSplitpointsBufferId(splitpointsBufferId)
, mSplitpointsCountsBufferId(splitpointsCountsBufferId)
, mChildCountsBufferId(childCountsBufferId)
, mLeftEstimatorParamsBufferId(leftEstimatorParamsBufferId)
, mRightEstimatorParamsBufferId(rightEstimatorParamsBufferId)
, mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
{}