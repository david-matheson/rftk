#pragma once

#include <vector>
#include <limits>
#include <cmath>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "bootstrap.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "FeatureSorter.h"
#include "AssignStreamStep.h"

// ----------------------------------------------------------------------------
//
// Finds the split point with the highest impurity for each feature.  It does
// this by sorting the feature values and walking the sorted values to find
// the split point with the highest impurity.  This uses a two stream
// walker where some points are used for calculating the impurity and some
// points are used for calculating the estimators
//
// ----------------------------------------------------------------------------
template <class ImpurityWalker>
class TwoStreamBestSplitpointsWalkingSortedStep : public PipelineStepI
{
public:
    TwoStreamBestSplitpointsWalkingSortedStep (const ImpurityWalker& impurityWalker,
                              const BufferId& streamTypeBufferId,
                              const BufferId& featureValues,
                              FeatureValueOrdering featureValueOrdering,
                              const int numberOfInBoundsDatapoints );
    virtual ~TwoStreamBestSplitpointsWalkingSortedStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    // Read only output buffer
    const BufferId ImpurityBufferId;
    const BufferId SplitpointBufferId;
    const BufferId SplitpointCountsBufferId;

    const BufferId ChildCountsEstimationBufferId;
    const BufferId LeftEstimationYsBufferId;
    const BufferId RightEstimationYsBufferId;
private:
    const ImpurityWalker mImpurityWalker;
    const BufferId mStreamTypeBufferId;
    const BufferId mFeatureValuesBufferId;
    const FeatureValueOrdering mFeatureValueOrdering;
    const int mNumberOfInBoundsDatapoints;
};


template <class ImpurityWalker>
TwoStreamBestSplitpointsWalkingSortedStep<ImpurityWalker>::TwoStreamBestSplitpointsWalkingSortedStep(const ImpurityWalker& impurityWalker,
                                                                                                  const BufferId& streamTypeBufferId,
                                                                                                  const BufferId& featureValues,
                                                                                                  FeatureValueOrdering featureValueOrdering,
                                                                                                  const int numberOfInBoundsDatapoints )
: PipelineStepI("TwoStreamBestSplitpointsWalkingSortedStep")
, ImpurityBufferId( GetBufferId("Impurity") )
, SplitpointBufferId( GetBufferId("Splitpoints") )
, SplitpointCountsBufferId( GetBufferId("SplitpointsCounts") )
, ChildCountsEstimationBufferId( GetBufferId("ChildCounts") )
, LeftEstimationYsBufferId( GetBufferId("LeftYs") )
, RightEstimationYsBufferId( GetBufferId("RightYs") )
, mImpurityWalker(impurityWalker)
, mStreamTypeBufferId(streamTypeBufferId)
, mFeatureValuesBufferId(featureValues)
, mFeatureValueOrdering(featureValueOrdering)
, mNumberOfInBoundsDatapoints(numberOfInBoundsDatapoints)
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
                                                              boost::mt19937& gen,
                                                              BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);

    // Bind input buffers
    ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::FeatureValue> >(mFeatureValuesBufferId));
    MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::FeatureValue> const& featureValues
           = readCollection.GetBuffer< MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::FeatureValue> >(mFeatureValuesBufferId);

    // Make a local non-const walker and bind it
    ImpurityWalker impurityWalker = mImpurityWalker;
    impurityWalker.Bind(readCollection);
    const int numberOfFeatures =  mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? featureValues.GetM() : featureValues.GetN();

    const VectorBufferTemplate<typename ImpurityWalker::BufferTypes::ParamsInteger>& streamTypes =
          readCollection.GetBuffer< VectorBufferTemplate<typename ImpurityWalker::BufferTypes::ParamsInteger> >(mStreamTypeBufferId);

    // Bind output buffers
    MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::ImpurityValue>& impurities
           = writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::ImpurityValue> >(ImpurityBufferId);
    impurities.Resize(numberOfFeatures,1);

    MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::FeatureValue>& splitpoints
           = writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::FeatureValue> >(SplitpointBufferId);
    splitpoints.Resize(numberOfFeatures,1);

    VectorBufferTemplate<typename ImpurityWalker::BufferTypes::Index>& splitpointCounts
           = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename ImpurityWalker::BufferTypes::Index> >(SplitpointCountsBufferId);
    splitpointCounts.Resize(numberOfFeatures);

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous>& childCounts
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> >(ChildCountsEstimationBufferId);
    childCounts.Resize(numberOfFeatures, 1, 2);

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous>& leftYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> >(LeftEstimationYsBufferId);
    leftYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous>& rightYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> >(RightEstimationYsBufferId);
    rightYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    for(int f=0; f<numberOfFeatures; f++)
    {
        FeatureSorter<typename ImpurityWalker::BufferTypes::FeatureValue> sorter(featureValues, mFeatureValueOrdering, f);
        sorter.Sort();

        const int numberOfSamples = sorter.GetNumberOfSamples();
        std::vector<int> inboundSamples(numberOfSamples);
        const int numberOfInBoundsSamples = std::min(numberOfSamples, mNumberOfInBoundsDatapoints);
        sampleWithOutReplacement(&inboundSamples[0], numberOfSamples, numberOfInBoundsSamples);

        typename ImpurityWalker::BufferTypes::FeatureValue boundsMin = std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::max();
        typename ImpurityWalker::BufferTypes::FeatureValue boundsMax = -std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::max();
        for(int sortedIndex=0; sortedIndex<numberOfSamples; sortedIndex++)
        {
            if(inboundSamples[sortedIndex]
                && streamTypes.Get(sorter.GetUnSortedIndex(sortedIndex)) == STREAM_STRUCTURE)
            {
                boundsMin = std::min(boundsMin, sorter.GetFeatureValue(sortedIndex));
                boundsMax = std::max(boundsMax, sorter.GetFeatureValue(sortedIndex));
            }
        }

        impurityWalker.Reset();

        bool hasValidSplit = false;

        typename ImpurityWalker::BufferTypes::ImpurityValue bestImpurity = -std::numeric_limits<typename ImpurityWalker::BufferTypes::ImpurityValue>::max();
        typename ImpurityWalker::BufferTypes::FeatureValue bestSplitpoint = std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::min();
        typename ImpurityWalker::BufferTypes::SufficientStatsContinuous bestLeftChildCounts = typename ImpurityWalker::BufferTypes::SufficientStatsContinuous(0);
        typename ImpurityWalker::BufferTypes::SufficientStatsContinuous bestRightChildCounts = typename ImpurityWalker::BufferTypes::SufficientStatsContinuous(0);
        VectorBufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> bestLeftYs(impurityWalker.GetYDim());
        VectorBufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> bestRightYs(impurityWalker.GetYDim());

        for(typename ImpurityWalker::BufferTypes::Index sortedIndex=0; sortedIndex<numberOfSamples-1; sortedIndex++)
        {
            const typename ImpurityWalker::BufferTypes::Index i = sorter.GetUnSortedIndex(sortedIndex);
            impurityWalker.MoveLeftToRight(i);

            const typename ImpurityWalker::BufferTypes::FeatureValue featureValue = sorter.GetFeatureValue(sortedIndex);
            const typename ImpurityWalker::BufferTypes::FeatureValue nextFeatureValue = sorter.GetFeatureValue(sortedIndex+1);
            const typename ImpurityWalker::BufferTypes::FeatureValue consecutiveFeatureDelta = nextFeatureValue - featureValue;
            if((std::abs(consecutiveFeatureDelta) > std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::epsilon())
              && (featureValue >= boundsMin)
              && (featureValue <= boundsMax)
              && streamTypes.Get(i) == STREAM_STRUCTURE
              && impurityWalker.Impurity() > bestImpurity)
            {
                hasValidSplit = true;
                bestImpurity = impurityWalker.Impurity();
                bestSplitpoint = sorter.GetFeatureValue(sortedIndex) + 0.5*consecutiveFeatureDelta;
                bestLeftChildCounts = impurityWalker.GetLeftEstimationChildCounts();
                bestRightChildCounts = impurityWalker.GetRightEstimationChildCounts();
                bestLeftYs = impurityWalker.GetLeftEstimationYs();
                bestRightYs = impurityWalker.GetRightEstimationYs();
            }
        }

        impurities.Set(f, 0, std::max(typename ImpurityWalker::BufferTypes::ImpurityValue(0), bestImpurity));
        splitpoints.Set(f, 0, bestSplitpoint);
        splitpointCounts.Set(f, hasValidSplit ? 1 : 0);
        childCounts.Set(f, 0, 0, bestLeftChildCounts);
        childCounts.Set(f, 0, 1, bestRightChildCounts);
        leftYs.SetRow(f, 0, bestLeftYs );
        rightYs.SetRow(f, 0, bestRightYs );
    }
}