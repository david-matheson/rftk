#pragma once


#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"


// ----------------------------------------------------------------------------
//
// Calculates the impurity for a set of stats
//
// ----------------------------------------------------------------------------
template <class Impurity>
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


template <class Impurity>
SplitpointsImpurity<Impurity>::SplitpointsImpurity(const BufferId& splitpointCountsBufferId,
                                                    const BufferId& childCountsBufferId,
                                                    const BufferId& leftStatsBufferId,
                                                    const BufferId& rightStatsBufferId)
: mSplitpointCountsBufferId(splitpointCountsBufferId)
, mChildCountsBufferId(childCountsBufferId)
, mLeftStatsBufferId(leftStatsBufferId)
, mRightStatsBufferId(rightStatsBufferId)
{}

template <class Impurity>
PipelineStepI* SplitpointsImpurity<Impurity>::Clone() const
{
    SplitpointsImpurity<Impurity>* clone = new SplitpointsImpurity<Impurity>(*this);
    return clone;
}

template <class Impurity>
void SplitpointsImpurity<Impurity>::ProcessStep(const BufferCollectionStack& readCollection,
                                                        BufferCollection& writeCollection,
                                                        boost::mt19937& gen) const
{
    UNUSED_PARAM(gen);

    const VectorBufferTemplate<typename Impurity::BufferTypes::Index>& splitpointsCounts =
          readCollection.GetBuffer< VectorBufferTemplate<typename Impurity::BufferTypes::Index> >(mSplitpointCountsBufferId);

    const Tensor3BufferTemplate<typename Impurity::BufferTypes::SufficientStatsContinuous>& childCounts =
          readCollection.GetBuffer< Tensor3BufferTemplate<typename Impurity::BufferTypes::SufficientStatsContinuous> >(mChildCountsBufferId);

    const Tensor3BufferTemplate<typename Impurity::BufferTypes::SufficientStatsContinuous>& leftStats =
          readCollection.GetBuffer< Tensor3BufferTemplate<typename Impurity::BufferTypes::SufficientStatsContinuous> >(mLeftStatsBufferId);

    const Tensor3BufferTemplate<typename Impurity::BufferTypes::SufficientStatsContinuous>& rightStats =
          readCollection.GetBuffer< Tensor3BufferTemplate<typename Impurity::BufferTypes::SufficientStatsContinuous> >(mRightStatsBufferId);

    const typename Impurity::BufferTypes::Index numberOfFeatures = childCounts.GetL();
    const typename Impurity::BufferTypes::Index maxNumberOfSplitpoints = childCounts.GetM();

    MatrixBufferTemplate<typename Impurity::BufferTypes::ImpurityValue>& impurities =
          writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename Impurity::BufferTypes::ImpurityValue> >(ImpurityBufferId);
    impurities.Resize(numberOfFeatures, maxNumberOfSplitpoints);

    Impurity ic;

    for(typename Impurity::BufferTypes::Index f=0; f<numberOfFeatures; f++)
    {
        const typename Impurity::BufferTypes::Index numberOfSplitpoints = splitpointsCounts.Get(f);
        for(typename Impurity::BufferTypes::Index s=0; s<numberOfSplitpoints; s++)
        {
            const typename Impurity::BufferTypes::ImpurityValue impurity = ic.Impurity(f, s, childCounts, leftStats, rightStats);
            impurities.Set(f, s, impurity);
        }
    }
}