#pragma once

#include <limits>
#include <cmath>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "FeatureSorter.h"

// ----------------------------------------------------------------------------
//
// Finds the split point with the highest impurity for each feature.  It does
// this by sorting the feature values and walking the sorted values to find
// the split point with the highest impurity.
//
// ----------------------------------------------------------------------------
template <class ImpurityWalker>
class TwoStreamBestSplitpointsWalkingSortedStep : public PipelineStepI
{
public:
    TwoStreamBestSplitpointsWalkingSortedStep (const ImpurityWalker& impurityWalker,
                              const BufferId& featureValues,
                              FeatureValueOrdering featureValueOrdering );
    virtual ~TwoStreamBestSplitpointsWalkingSortedStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    // Read only output buffer
    const BufferId ImpurityBufferId;
    const BufferId SplitpointBufferId;
    const BufferId SplitpointCountsBufferId;

    const BufferId ChildCountsEstimationBufferId;
    const BufferId LeftEstimationYsBufferId;
    const BufferId RightEstimationYsBufferId;
private:
    const ImpurityWalker mImpurityWalker;
    const BufferId mFeatureValuesBufferId;
    const FeatureValueOrdering mFeatureValueOrdering;

};


template <class ImpurityWalker>
TwoStreamBestSplitpointsWalkingSortedStep<ImpurityWalker>::TwoStreamBestSplitpointsWalkingSortedStep(const ImpurityWalker& impurityWalker,
                                                                      const BufferId& featureValues,
                                                                      FeatureValueOrdering featureValueOrdering )
: ImpurityBufferId( GetBufferId("Impurity") )
, SplitpointBufferId( GetBufferId("Splitpoints") )
, SplitpointCountsBufferId( GetBufferId("SplitpointsCounts") )
, ChildCountsEstimationBufferId( GetBufferId("ChildCounts") )
, LeftEstimationYsBufferId( GetBufferId("LeftYs") )
, RightEstimationYsBufferId( GetBufferId("RightYs") )
, mImpurityWalker(impurityWalker)
, mFeatureValuesBufferId(featureValues)
, mFeatureValueOrdering(featureValueOrdering)
{}

template <class ImpurityWalker>
TwoStreamBestSplitpointsWalkingSortedStep<ImpurityWalker>::~TwoStreamBestSplitpointsWalkingSortedStep()
{}

template <class ImpurityWalker>
PipelineStepI* TwoStreamBestSplitpointsWalkingSortedStep<ImpurityWalker>::Clone() const
{
    TwoStreamBestSplitpointsWalkingSortedStep* clone = new TwoStreamBestSplitpointsWalkingSortedStep<ImpurityWalker>(*this);
    return clone;
}

template <class ImpurityWalker>
void TwoStreamBestSplitpointsWalkingSortedStep<ImpurityWalker>::ProcessStep(const BufferCollectionStack& readCollection,
                                                              BufferCollection& writeCollection,
                                                              boost::mt19937& gen) const
{
    UNUSED_PARAM(gen);

    // Bind input buffers
    ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<typename ImpurityWalker::Float> >(mFeatureValuesBufferId));
    MatrixBufferTemplate<typename ImpurityWalker::Float> const& featureValues
           = readCollection.GetBuffer< MatrixBufferTemplate<typename ImpurityWalker::Float> >(mFeatureValuesBufferId);

    // Make a local non-const walker and bind it
    ImpurityWalker impurityWalker = mImpurityWalker;
    impurityWalker.Bind(readCollection);
    const int numberOfFeatures =  mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? featureValues.GetM() : featureValues.GetN();

    // Bind output buffers
    MatrixBufferTemplate<typename ImpurityWalker::Float>& impurities
           = writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename ImpurityWalker::Float> >(ImpurityBufferId);
    impurities.Resize(numberOfFeatures,1);

    MatrixBufferTemplate<typename ImpurityWalker::Float>& thresholds
           = writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename ImpurityWalker::Float> >(SplitpointBufferId);
    thresholds.Resize(numberOfFeatures,1);

    VectorBufferTemplate<typename ImpurityWalker::Int>& thresholdCounts
           = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename ImpurityWalker::Int> >(SplitpointCountsBufferId);
    thresholdCounts.Resize(numberOfFeatures);

    Tensor3BufferTemplate<typename ImpurityWalker::Float>& childCounts
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::Float> >(ChildCountsEstimationBufferId);
    childCounts.Resize(numberOfFeatures, 1, 2);

    Tensor3BufferTemplate<typename ImpurityWalker::Float>& leftYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::Float> >(LeftEstimationYsBufferId);
    leftYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    Tensor3BufferTemplate<typename ImpurityWalker::Float>& rightYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::Float> >(RightEstimationYsBufferId);
    rightYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    for(int f=0; f<numberOfFeatures; f++)
    {
        impurityWalker.Reset();

        typename ImpurityWalker::Float bestImpurity = std::numeric_limits<typename ImpurityWalker::Float>::min();
        typename ImpurityWalker::Float bestThreshold = std::numeric_limits<typename ImpurityWalker::Float>::min();
        typename ImpurityWalker::Float bestLeftChildCounts = typename ImpurityWalker::Float(0);
        typename ImpurityWalker::Float bestRightChildCounts = typename ImpurityWalker::Float(0);
        VectorBufferTemplate<typename ImpurityWalker::Float> bestLeftYs(impurityWalker.GetYDim());
        VectorBufferTemplate<typename ImpurityWalker::Float> bestRightYs(impurityWalker.GetYDim());

        FeatureSorter<typename ImpurityWalker::Float> sorter(featureValues, mFeatureValueOrdering, f);
        sorter.Sort();
        for(int sortedIndex=0; sortedIndex<sorter.GetNumberOfSamples()-1; sortedIndex++)
        {
            const int i = sorter.GetUnSortedIndex(sortedIndex);

            impurityWalker.MoveLeftToRight(i);

            const typename ImpurityWalker::Float consecutiveFeatureDelta = sorter.GetFeatureValue(sortedIndex+1) - sorter.GetFeatureValue(sortedIndex);
            if((std::abs(consecutiveFeatureDelta) > std::numeric_limits<typename ImpurityWalker::Float>::epsilon())
              && impurityWalker.Impurity() > bestImpurity)
            {
                bestImpurity = impurityWalker.Impurity();
                bestThreshold = sorter.GetFeatureValue(sortedIndex) + 0.5*consecutiveFeatureDelta;
                bestLeftChildCounts = impurityWalker.GetLeftEstimationChildCounts();
                bestRightChildCounts = impurityWalker.GetRightEstimationChildCounts();
                bestLeftYs = impurityWalker.GetLeftEstimationYs();
                bestRightYs = impurityWalker.GetRightEstimationYs();
            }
        }

        impurities.Set(f, 0, bestImpurity);
        thresholds.Set(f, 0, bestThreshold);
        thresholdCounts.Set(f, 1);
        childCounts.Set(f, 0, 0, bestLeftChildCounts);
        childCounts.Set(f, 0, 1, bestRightChildCounts);
        leftYs.SetRow(f, 0, bestLeftYs );
        rightYs.SetRow(f, 0, bestRightYs );
    }
}