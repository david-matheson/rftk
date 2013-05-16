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
                                boost::mt19937& gen) const;

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
                                                        boost::mt19937& gen) const
{
    UNUSED_PARAM(gen);

    typename StatsUpdater::BindedStatUpdater bindedStatUpdater = mStatsUpdater.Bind(readCollection);

    const MatrixBufferTemplate<typename StatsUpdater::Float>& splitpoints =
          readCollection.GetBuffer< MatrixBufferTemplate<typename StatsUpdater::Float> >(mSplitpointsBufferId);

    const VectorBufferTemplate<typename StatsUpdater::Int>& splitpointsCounts =
          readCollection.GetBuffer< VectorBufferTemplate<typename StatsUpdater::Int> >(mSplitpointCountsBufferId);

    const MatrixBufferTemplate<typename StatsUpdater::Float>& featureValues =
          readCollection.GetBuffer< MatrixBufferTemplate<typename StatsUpdater::Float> >(mFeatureValuesBufferId);

    Tensor3BufferTemplate<typename StatsUpdater::Float>& childCounts =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::Float> >(ChildCountsBufferId);
    childCounts.Resize(splitpoints.GetM(), splitpoints.GetN(), 2);

    Tensor3BufferTemplate<typename StatsUpdater::Float>& leftStats =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::Float> >(LeftStatsBufferId);
    leftStats.Resize(splitpoints.GetM(), splitpoints.GetN(), mStatsUpdater.GetNumberOfClasses());

    Tensor3BufferTemplate<typename StatsUpdater::Float>& rightStats =
          writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename StatsUpdater::Float> >(RightStatsBufferId);
    rightStats.Resize(splitpoints.GetM(), splitpoints.GetN(), mStatsUpdater.GetNumberOfClasses());

    for(int c=0; c<featureValues.GetM(); c++)
    {
        for(int r=0; r<featureValues.GetN(); r++)
        {
            const typename StatsUpdater::Float featureValue = featureValues.Get(c,r);
            const int feature = mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? c : r;
            const int sample = mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? r : c;

            for(int splitpoint=0; splitpoint<splitpointsCounts.Get(feature); splitpoint++)
            {
                const typename StatsUpdater::Float splitPointValue = splitpoints.Get(feature, splitpoint);
                if( featureValue > splitPointValue )
                {
                    typename StatsUpdater::Float counts = childCounts.Get(feature, splitpoint, LEFT_CHILD);
                    bindedStatUpdater.UpdateStats(counts, leftStats, feature, splitpoint, sample );
                    childCounts.Set(feature, splitpoint, 0, counts);
                }
                else
                {
                    typename StatsUpdater::Float counts = childCounts.Get(feature, splitpoint, RIGHT_CHILD);
                    bindedStatUpdater.UpdateStats(counts, rightStats, feature, splitpoint, sample );
                    childCounts.Set(feature, splitpoint, 1, counts);
                }
            }
        }
    }
}
