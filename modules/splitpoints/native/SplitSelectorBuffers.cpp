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
, mFeatureIndexer(NULL)
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
                                            FeatureValueOrdering ordering,
                                            const FeatureInfoLoggerI* featureIndexer)
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
, mFeatureIndexer( featureIndexer != NULL ? featureIndexer->CloneFeatureInfoLoggerI() : NULL)
{}

SplitSelectorBuffers::~SplitSelectorBuffers()
{
	if(mFeatureIndexer != NULL)
	{
		delete mFeatureIndexer;	
	}
}

SplitSelectorBuffers::SplitSelectorBuffers(const SplitSelectorBuffers& other)
: mImpurityBufferId(other.mImpurityBufferId)
, mSplitpointsBufferId(other.mSplitpointsBufferId)
, mSplitpointsCountsBufferId(other.mSplitpointsCountsBufferId)
, mChildCountsBufferId(other.mChildCountsBufferId)
, mLeftEstimatorParamsBufferId(other.mLeftEstimatorParamsBufferId)
, mRightEstimatorParamsBufferId(other.mRightEstimatorParamsBufferId)
, mFloatParamsBufferId(other.mFloatParamsBufferId) 
, mIntParamsBufferId(other.mIntParamsBufferId)
, mFeatureValuesBufferId(other.mFeatureValuesBufferId)
, mOrdering(other.mOrdering)
, mFeatureIndexer( other.mFeatureIndexer != NULL ? other.mFeatureIndexer->CloneFeatureInfoLoggerI() : NULL )
{}

SplitSelectorBuffers& SplitSelectorBuffers::operator=( const SplitSelectorBuffers& rhs )
{
	this->mImpurityBufferId = rhs.mImpurityBufferId;
	this->mSplitpointsBufferId = rhs.mSplitpointsBufferId;
	this->mSplitpointsCountsBufferId = rhs.mSplitpointsCountsBufferId;
	this->mChildCountsBufferId = rhs.mChildCountsBufferId;
	this->mLeftEstimatorParamsBufferId = rhs.mLeftEstimatorParamsBufferId;
	this->mRightEstimatorParamsBufferId = rhs.mRightEstimatorParamsBufferId;
	this->mFloatParamsBufferId = rhs.mFloatParamsBufferId;
	this->mIntParamsBufferId = rhs.mIntParamsBufferId;
	this->mFeatureValuesBufferId = rhs.mFeatureValuesBufferId;
	this->mOrdering = rhs.mOrdering;

	if(this->mFeatureIndexer != NULL)
	{
		delete this->mFeatureIndexer;
		this->mFeatureIndexer = NULL;	

	}
	if(rhs.mFeatureIndexer != NULL)
	{
		this->mFeatureIndexer = rhs.mFeatureIndexer->CloneFeatureInfoLoggerI();
	}
	return *this;
}