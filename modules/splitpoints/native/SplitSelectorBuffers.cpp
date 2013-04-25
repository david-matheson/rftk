#include "SplitSelectorBuffers.h"

SplitSelectorBuffers::SplitSelectorBuffers(const UniqueBufferId::BufferId& impurityBufferId,
                                            const UniqueBufferId::BufferId& splitpointsBufferId,
                                            const UniqueBufferId::BufferId& splitpointsCountsBufferId,
                                            const UniqueBufferId::BufferId& childCountsBufferId,
                                            const UniqueBufferId::BufferId& leftEstimatorParamsBufferId,
                                            const UniqueBufferId::BufferId& rightEstimatorParamsBufferId,
                                            const UniqueBufferId::BufferId& floatParamsBufferId,
                                            const UniqueBufferId::BufferId& intParamsBufferId,
                                            const UniqueBufferId::BufferId& featureValuesBufferId,
                                            FeatureValueOrdering ordering,
                                            const UniqueBufferId::BufferId& indicesBufferId)
: mImpurityBufferId(impurityBufferId)
, mSplitpointsBufferId(splitpointsBufferId)
, mSplitpointsCountsBufferId(splitpointsCountsBufferId)
, mChildCountsBufferId(childCountsBufferId)
, mLeftEstimatorParamsBufferId(leftEstimatorParamsBufferId)
, mRightEstimatorParamsBufferId(rightEstimatorParamsBufferId)
, mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
, mFeatureValuesBufferId(featureValuesBufferId)
, mOrdering(ordering)
, mIndicesBufferId(indicesBufferId)
{}