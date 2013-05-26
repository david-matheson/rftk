#include "UniqueBufferId.h"
#include "SplitSelectorBuffers.h"

SplitSelectorBuffers::SplitSelectorBuffers()
: mImpurityBufferId()
, mSplitpointsBufferId()
, mSplitpointsCountsBufferId()
, mChildCountsBufferId()
, mLeftEstimatorParamsBufferId()
, mRightEstimatorParamsBufferId()
, mFloatParamsBufferId() 
, mIntParamsBufferId()
, mFeatureValuesBufferId()
, mOrdering(FEATURES_BY_DATAPOINTS)
{}

SplitSelectorBuffers::SplitSelectorBuffers(const BufferId& impurityBufferId,
                                            const BufferId& splitpointsBufferId,
                                            const BufferId& splitpointsCountsBufferId,
                                            const BufferId& childCountsBufferId,
                                            const BufferId& leftEstimatorParamsBufferId,
                                            const BufferId& rightEstimatorParamsBufferId,
                                            const BufferId& floatParamsBufferId,
                                            const BufferId& intParamsBufferId,
                                            const BufferId& featureValuesBufferId,
                                            FeatureValueOrdering ordering)
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
{}

