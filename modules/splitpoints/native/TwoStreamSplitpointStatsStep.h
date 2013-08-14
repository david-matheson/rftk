#pragma once


#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "FeatureExtractorStep.h"
#include "UniqueBufferId.h"
#include "Constants.h"
#include "AssignStreamStep.h"

// ----------------------------------------------------------------------------
//
// Updates the split statistics for #features X #splitpoints for impurity and
// estimator streams
//
// ----------------------------------------------------------------------------
template <class StatsUpdater>
class TwoStreamSplitpointStatsStep : public PipelineStepI
{
public:
    TwoStreamSplitpointStatsStep(const BufferId& splitpointsBufferId,
                            const BufferId& splitpointCountsBufferId,
                            const BufferId& streamTypeBufferId,
                            const BufferId& featureValuesBufferId,
                            FeatureValueOrdering featureValueOrdering,
                            const StatsUpdater& statsUpdater);

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    const BufferId ChildCountsImpurityBufferId;
    const BufferId LeftImpurityStatsBufferId;
    const BufferId RightImpurityStatsBufferId;

    const BufferId ChildCountsEstimatorBufferId;
    const BufferId LeftEstimatorStatsBufferId;
    const BufferId RightEstimatorStatsBufferId;

private:
    const BufferId mSplitpointsBufferId;
    const BufferId mSplitpointCountsBufferId;
    const BufferId mStreamTypeBufferId;
    const BufferId mFeatureValuesBufferId;
    const FeatureValueOrdering mFeatureValueOrdering;
    StatsUpdater mStatsUpdater;
};

template <class StatsUpdater>
TwoStreamSplitpointStatsStep<StatsUpdater>::TwoStreamSplitpointStatsStep(const BufferId& splitpointsBufferId,
                                                              const BufferId& splitpointCountsBufferId,
                                                              const BufferId& streamTypeBufferId,
                                                              const BufferId& featureValuesBufferId,
                                                              FeatureValueOrdering featureValueOrdering,
                                                              const StatsUpdater& statsUpdater)
: ChildCountsImpurityBufferId(GetBufferId("ChildCountsImpurity"))
, LeftImpurityStatsBufferId(GetBufferId("LeftImpurityStats"))
, RightImpurityStatsBufferId(GetBufferId("RightImpurityStats"))
, ChildCountsEstimatorBufferId(GetBufferId("ChildCountsEstimator"))
, LeftEstimatorStatsBufferId(GetBufferId("LeftEstimatorStats"))
, RightEstimatorStatsBufferId(GetBufferId("RightEstimatorStats"))
, mSplitpointsBufferId(splitpointsBufferId)
, mSplitpointCountsBufferId(splitpointCountsBufferId)
, mStreamTypeBufferId(streamTypeBufferId)
, mFeatureValuesBufferId(featureValuesBufferId)
, mFeatureValueOrdering(featureValueOrdering)
, mStatsUpdater(statsUpdater)
{
}

template <class StatsUpdater>
PipelineStepI* TwoStreamSplitpointStatsStep<StatsUpdater>::Clone() const
{
    TwoStreamSplitpointStatsStep<StatsUpdater>* clone = new TwoStreamSplitpointStatsStep<StatsUpdater>(*this);
    return clone;
}

template <class StatsUpdater>
void TwoStreamSplitpointStatsStep<StatsUpdater>::ProcessStep(const BufferCollectionStack& readCollection,
                                                        BufferCollection& writeCollection,
                                                        boost::mt19937& gen) const
{
    UNUSED_PARAM(gen);

    typename StatsUpdater::BindedStatUpdater bindedStatUpdater = mStatsUpdater.Bind(readCollection);

    const MatrixBufferTemplate<typename StatsUpdater::BufferTypes::FeatureValue>& splitpoints =
          readCollection.GetBuffer< MatrixBufferTemplate<typename StatsUpdater::BufferTypes::FeatureValue> >(mSplitpointsBufferId);

    const VectorBufferTemplate<typename StatsUpdater::BufferTypes::Index>& splitpointsCounts =
          readCollection.GetBuffer< VectorBufferTemplate<typename StatsUpdater::BufferTypes::Index> >(mSplitpointCountsBufferId);

    const VectorBufferTemplate<typename StatsUpdater::BufferTypes::ParamsInteger>& streamTypes =
          readCollection.GetBuffer< VectorBufferTemplate<typename StatsUpdater::BufferTypes::ParamsInteger> >(mStreamTypeBufferId);

    const MatrixBufferTemplate<typename StatsUpdater::BufferTypes::FeatureValue>& featureValues =
          readCollection.GetBuffer< MatrixBufferTemplate<typename StatsUpdater::BufferTypes::FeatureValue> >(mFeatureValuesBufferId);

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::DatapointCounts>& childCountsImpurity =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::DatapointCounts> >(ChildCountsImpurityBufferId);
    childCountsImpurity.Resize(splitpoints.GetM(), splitpoints.GetN(), 2);

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous>& leftImpurityStats =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous> >(LeftImpurityStatsBufferId);
    leftImpurityStats.Resize(splitpoints.GetM(), splitpoints.GetN(), mStatsUpdater.GetDimension());

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous>& rightImpurityStats =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous> >(RightImpurityStatsBufferId);
    rightImpurityStats.Resize(splitpoints.GetM(), splitpoints.GetN(), mStatsUpdater.GetDimension());

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::DatapointCounts>& childCountsEstimator =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::DatapointCounts> >(ChildCountsEstimatorBufferId);
    childCountsEstimator.Resize(splitpoints.GetM(), splitpoints.GetN(), 2);

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous>& leftEstimatorStats =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous> >(LeftEstimatorStatsBufferId);
    leftEstimatorStats.Resize(splitpoints.GetM(), splitpoints.GetN(), mStatsUpdater.GetDimension());

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous>& rightEstimatorStats =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous> >(RightEstimatorStatsBufferId);
    rightEstimatorStats.Resize(splitpoints.GetM(), splitpoints.GetN(), mStatsUpdater.GetDimension());

    for(typename StatsUpdater::BufferTypes::Index c=0; c<featureValues.GetM(); c++)
    {
        for(typename StatsUpdater::BufferTypes::Index r=0; r<featureValues.GetN(); r++)
        {
            const typename StatsUpdater::BufferTypes::FeatureValue featureValue = featureValues.Get(c,r);
            const typename StatsUpdater::BufferTypes::Index feature = mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? c : r;
            const typename StatsUpdater::BufferTypes::Index sample = mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? r : c;
            const typename StatsUpdater::BufferTypes::Index streamType = streamTypes.Get(sample);

            for(typename StatsUpdater::BufferTypes::Index splitpoint=0; splitpoint<splitpointsCounts.Get(feature); splitpoint++)
            {
                const typename StatsUpdater::BufferTypes::FeatureValue splitPointValue = splitpoints.Get(feature, splitpoint);
                if( featureValue > splitPointValue )
                {
                    if(streamType == STREAM_STRUCTURE)
                    {
                        typename StatsUpdater::BufferTypes::DatapointCounts counts = childCountsImpurity.Get(feature, splitpoint, LEFT_CHILD);
                        bindedStatUpdater.UpdateStats(counts, leftImpurityStats, feature, splitpoint, sample );
                        childCountsImpurity.Set(feature, splitpoint, LEFT_CHILD, counts);
                    }
                    else if(streamType == STREAM_ESTIMATION)
                    {
                        typename StatsUpdater::BufferTypes::DatapointCounts counts = childCountsEstimator.Get(feature, splitpoint, LEFT_CHILD);
                        bindedStatUpdater.UpdateStats(counts, leftEstimatorStats, feature, splitpoint, sample );
                        childCountsEstimator.Set(feature, splitpoint, LEFT_CHILD, counts);
                    }
                }
                else
                {
                    if(streamType == STREAM_STRUCTURE)
                    {
                        typename StatsUpdater::BufferTypes::DatapointCounts counts = childCountsImpurity.Get(feature, splitpoint, RIGHT_CHILD);
                        bindedStatUpdater.UpdateStats(counts, rightImpurityStats, feature, splitpoint, sample );
                        childCountsImpurity.Set(feature, splitpoint, RIGHT_CHILD, counts);
                    }
                    else if(streamType == STREAM_ESTIMATION)
                    {
                        typename StatsUpdater::BufferTypes::DatapointCounts counts = childCountsEstimator.Get(feature, splitpoint, RIGHT_CHILD);
                        bindedStatUpdater.UpdateStats(counts, rightEstimatorStats, feature, splitpoint, sample );
                        childCountsEstimator.Set(feature, splitpoint, RIGHT_CHILD, counts);
                    }
                }
            }
        }
    }
}
