#pragma once


#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "FeatureExtractorStep.h"
#include "UniqueBufferId.h"
#include "Constants.h"

// ----------------------------------------------------------------------------
//
// Updates the split statistics #features X #splitpoints
//
// ----------------------------------------------------------------------------
template <class StatsUpdater>
class SplitpointStatsStep : public PipelineStepI
{
public:
    SplitpointStatsStep(const BufferId& splitpointsBufferId,
                            const BufferId& splitpointCountsBufferId,
                            const BufferId& featureValuesBufferId,
                            FeatureValueOrdering featureValueOrdering,
                            const StatsUpdater& statsUpdater);

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    const BufferId ChildCountsBufferId;
    const BufferId LeftStatsBufferId;
    const BufferId RightStatsBufferId;

private:
    const BufferId mSplitpointsBufferId;
    const BufferId mSplitpointCountsBufferId;
    const BufferId mFeatureValuesBufferId;
    const FeatureValueOrdering mFeatureValueOrdering;
    StatsUpdater mStatsUpdater;
};

template <class StatsUpdater>
SplitpointStatsStep<StatsUpdater>::SplitpointStatsStep(const BufferId& splitpointsBufferId,
                                                              const BufferId& splitpointCountsBufferId,
                                                              const BufferId& featureValuesBufferId,
                                                              FeatureValueOrdering featureValueOrdering,
                                                              const StatsUpdater& statsUpdater)
: ChildCountsBufferId(GetBufferId("ChildCounts"))
, LeftStatsBufferId(GetBufferId("LeftStats"))
, RightStatsBufferId(GetBufferId("RightStats"))
, mSplitpointsBufferId(splitpointsBufferId)
, mSplitpointCountsBufferId(splitpointCountsBufferId)
, mFeatureValuesBufferId(featureValuesBufferId)
, mFeatureValueOrdering(featureValueOrdering)
, mStatsUpdater(statsUpdater)
{
}

template <class StatsUpdater>
PipelineStepI* SplitpointStatsStep<StatsUpdater>::Clone() const
{
    SplitpointStatsStep<StatsUpdater>* clone = new SplitpointStatsStep<StatsUpdater>(*this);
    return clone;
}

template <class StatsUpdater>
void SplitpointStatsStep<StatsUpdater>::ProcessStep(const BufferCollectionStack& readCollection,
                                                        BufferCollection& writeCollection,
                                                        boost::mt19937& gen,
                                                        BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);

    typename StatsUpdater::BindedStatUpdater bindedStatUpdater = mStatsUpdater.Bind(readCollection);

    const MatrixBufferTemplate<typename StatsUpdater::BufferTypes::FeatureValue>& splitpoints =
          readCollection.GetBuffer< MatrixBufferTemplate<typename StatsUpdater::BufferTypes::FeatureValue> >(mSplitpointsBufferId);

    const VectorBufferTemplate<typename StatsUpdater::BufferTypes::Index>& splitpointsCounts =
          readCollection.GetBuffer< VectorBufferTemplate<typename StatsUpdater::BufferTypes::Index> >(mSplitpointCountsBufferId);

    const MatrixBufferTemplate<typename StatsUpdater::BufferTypes::FeatureValue>& featureValues =
          readCollection.GetBuffer< MatrixBufferTemplate<typename StatsUpdater::BufferTypes::FeatureValue> >(mFeatureValuesBufferId);

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::DatapointCounts>& childCounts =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::DatapointCounts> >(ChildCountsBufferId);
    childCounts.Resize(splitpoints.GetM(), splitpoints.GetN(), 2);

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous>& leftStats =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous> >(LeftStatsBufferId);
    leftStats.Resize(splitpoints.GetM(), splitpoints.GetN(), mStatsUpdater.GetDimension());

    Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous>& rightStats =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::BufferTypes::SufficientStatsContinuous> >(RightStatsBufferId);
    rightStats.Resize(splitpoints.GetM(), splitpoints.GetN(), mStatsUpdater.GetDimension());

    for(typename StatsUpdater::BufferTypes::Index c=0; c<featureValues.GetM(); c++)
    {
        for(typename StatsUpdater::BufferTypes::Index r=0; r<featureValues.GetN(); r++)
        {
            const typename StatsUpdater::BufferTypes::FeatureValue featureValue = featureValues.Get(c,r);
            const typename StatsUpdater::BufferTypes::Index feature = mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? c : r;
            const typename StatsUpdater::BufferTypes::Index sample = mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? r : c;

            for(typename StatsUpdater::BufferTypes::Index splitpoint=0; splitpoint<splitpointsCounts.Get(feature); splitpoint++)
            {
                const typename StatsUpdater::BufferTypes::FeatureValue splitPointValue = splitpoints.Get(feature, splitpoint);
                if( featureValue > splitPointValue )
                {
                    typename StatsUpdater::BufferTypes::DatapointCounts counts = childCounts.Get(feature, splitpoint, LEFT_CHILD);
                    bindedStatUpdater.UpdateStats(counts, leftStats, feature, splitpoint, sample );
                    childCounts.Set(feature, splitpoint, 0, counts);
                }
                else
                {
                    typename StatsUpdater::BufferTypes::DatapointCounts counts = childCounts.Get(feature, splitpoint, RIGHT_CHILD);
                    bindedStatUpdater.UpdateStats(counts, rightStats, feature, splitpoint, sample );
                    childCounts.Set(feature, splitpoint, 1, counts);
                }
            }
        }
    }
}
