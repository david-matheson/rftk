#pragma once


#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

#include "FeatureExtractorStep.h"

// ----------------------------------------------------------------------------
//
// Calculates the impurity for a set of stats
//
// ----------------------------------------------------------------------------
template <class Impurity, class IntType>
class SplitpointsImpurity : public PipelineStepI
{
public:
    SplitpointsImpurity(const BufferId& splitpointCountsBufferId,
                        const BufferId& childCountsBufferId,
                        const BufferId& leftStatsBufferId,
                        const BufferId& rightStatsBufferId);

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    const BufferId ImpurityBufferId;
private:
    const BufferId mSplitpointCountsBufferId;
    const BufferId mChildCountsBufferId;
    const BufferId mLeftStatsBufferId;
    const BufferId mRightStatsBufferId;
};


template <class Impurity, class IntType>
SplitpointsImpurity<Impurity, IntType>::SplitpointsImpurity(const BufferId& splitpointCountsBufferId,
                                                    const BufferId& childCountsBufferId,
                                                    const BufferId& leftStatsBufferId,
                                                    const BufferId& rightStatsBufferId)
: mSplitpointCountsBufferId(splitpointCountsBufferId)
, mChildCountsBufferId(childCountsBufferId)
, mLeftStatsBufferId(leftStatsBufferId)
, mRightStatsBufferId(rightStatsBufferId)
{}

template <class Impurity, class IntType>
PipelineStepI* SplitpointsImpurity<Impurity, IntType>::Clone() const
{
    SplitpointsImpurity<Impurity, IntType>* clone = new SplitpointsImpurity<Impurity, IntType>(*this);
    return clone;
}

template <class Impurity, class IntType>
void SplitpointsImpurity<Impurity, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                        BufferCollection& writeCollection,
                                                        boost::mt19937& gen) const
{
    UNUSED_PARAM(gen);

    const VectorBufferTemplate<IntType>& splitpointsCounts =
          readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mSplitpointCountsBufferId);

    const Tensor3BufferTemplate<typename Impurity::Float>& childCounts =
          readCollection.GetBuffer< Tensor3BufferTemplate<typename Impurity::Float> >(mChildCountsBufferId);

    const Tensor3BufferTemplate<typename Impurity::Float>& leftStats =
          readCollection.GetBuffer< Tensor3BufferTemplate<typename Impurity::Float> >(mLeftStatsBufferId);

    const Tensor3BufferTemplate<typename Impurity::Float>& rightStats =
          readCollection.GetBuffer< Tensor3BufferTemplate<typename Impurity::Float> >(mRightStatsBufferId);

    const int numberOfFeatures = childCounts.GetL();
    const int maxNumberOfSplitpoints = childCounts.GetM();

    MatrixBufferTemplate<typename Impurity::Float>& impurities =
          writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename Impurity::Float> >(ImpurityBufferId);
    impurities.Resize(numberOfFeatures, maxNumberOfSplitpoints);

    Impurity ic;

    for(int f=0; f<numberOfFeatures; f++)
    {
        const int numberOfSplitpoints = splitpointsCounts.Get(f);
        for(int s=0; s<numberOfSplitpoints; s++)
        {
            const typename Impurity::Float impurity = ic.Impurity(f, s, childCounts, leftStats, rightStats);
            impurities.Set(f, s, impurity);
        }
    }
}