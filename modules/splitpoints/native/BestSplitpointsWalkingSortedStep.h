#pragma once

#include <limits>
#include <cmath>

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>

#include "bootstrap.h"
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
enum WalkingSortedSplitpointLocation
{
    AT_MIDPOINT,
    AT_DATAPOINT,
    UNIFORM_AT_GAP
};

template <class ImpurityWalker>
class BestSplitpointsWalkingSortedStep : public PipelineStepI
{
public:
    BestSplitpointsWalkingSortedStep (const ImpurityWalker& impurityWalker,
                              const BufferId& featureValues,
                              FeatureValueOrdering featureValueOrdering);
    BestSplitpointsWalkingSortedStep (const ImpurityWalker& impurityWalker,
                              const BufferId& featureValues,
                              FeatureValueOrdering featureValueOrdering,
                              WalkingSortedSplitpointLocation splitpointLocation );
    BestSplitpointsWalkingSortedStep (const ImpurityWalker& impurityWalker,
                              const BufferId& featureValues,
                              FeatureValueOrdering featureValueOrdering,
                              WalkingSortedSplitpointLocation splitpointLocation,
                              const int numberOfInBoundsDatapoints );
    virtual ~BestSplitpointsWalkingSortedStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    // Read only output buffer
    const BufferId ImpurityBufferId;
    const BufferId SplitpointBufferId;
    const BufferId SplitpointCountsBufferId;
    const BufferId ChildCountsBufferId;
    const BufferId LeftYsBufferId;
    const BufferId RightYsBufferId;
private:
    const ImpurityWalker mImpurityWalker;
    const BufferId mFeatureValuesBufferId;
    const FeatureValueOrdering mFeatureValueOrdering;
    const WalkingSortedSplitpointLocation mSplitpointLocation;
    const int mNumberOfInBoundsDatapoints;

};


template <class ImpurityWalker>
BestSplitpointsWalkingSortedStep<ImpurityWalker>::BestSplitpointsWalkingSortedStep(const ImpurityWalker& impurityWalker,
                                                                      const BufferId& featureValues,
                                                                      FeatureValueOrdering featureValueOrdering,
                                                                      WalkingSortedSplitpointLocation splitpointLocation )
: PipelineStepI("BestSplitpointsWalkingSortedStep")
, ImpurityBufferId( GetBufferId("Impurity") )
, SplitpointBufferId( GetBufferId("Splitpoints") )
, SplitpointCountsBufferId( GetBufferId("SplitpointsCounts") )
, ChildCountsBufferId( GetBufferId("ChildCounts") )
, LeftYsBufferId( GetBufferId("LeftYs") )
, RightYsBufferId( GetBufferId("RightYs") )
, mImpurityWalker(impurityWalker)
, mFeatureValuesBufferId(featureValues)
, mFeatureValueOrdering(featureValueOrdering)
, mSplitpointLocation(splitpointLocation)
, mNumberOfInBoundsDatapoints(0)
{}

template <class ImpurityWalker>
BestSplitpointsWalkingSortedStep<ImpurityWalker>::BestSplitpointsWalkingSortedStep(const ImpurityWalker& impurityWalker,
                                                                      const BufferId& featureValues,
                                                                      FeatureValueOrdering featureValueOrdering )
: PipelineStepI("BestSplitpointsWalkingSortedStep")
, ImpurityBufferId( GetBufferId("Impurity") )
, SplitpointBufferId( GetBufferId("Splitpoints") )
, SplitpointCountsBufferId( GetBufferId("SplitpointsCounts") )
, ChildCountsBufferId( GetBufferId("ChildCounts") )
, LeftYsBufferId( GetBufferId("LeftYs") )
, RightYsBufferId( GetBufferId("RightYs") )
, mImpurityWalker(impurityWalker)
, mFeatureValuesBufferId(featureValues)
, mFeatureValueOrdering(featureValueOrdering)
, mSplitpointLocation(AT_MIDPOINT)
, mNumberOfInBoundsDatapoints(0)
{}


template <class ImpurityWalker>
BestSplitpointsWalkingSortedStep<ImpurityWalker>::BestSplitpointsWalkingSortedStep(const ImpurityWalker& impurityWalker,
                                                                      const BufferId& featureValues,
                                                                      FeatureValueOrdering featureValueOrdering,
                                                                      WalkingSortedSplitpointLocation splitpointLocation,
                                                                      const int numberOfInBoundsDatapoints)
: PipelineStepI("BestSplitpointsWalkingSortedStep")
, ImpurityBufferId( GetBufferId("Impurity") )
, SplitpointBufferId( GetBufferId("Splitpoints") )
, SplitpointCountsBufferId( GetBufferId("SplitpointsCounts") )
, ChildCountsBufferId( GetBufferId("ChildCounts") )
, LeftYsBufferId( GetBufferId("LeftYs") )
, RightYsBufferId( GetBufferId("RightYs") )
, mImpurityWalker(impurityWalker)
, mFeatureValuesBufferId(featureValues)
, mFeatureValueOrdering(featureValueOrdering)
, mSplitpointLocation(splitpointLocation)
, mNumberOfInBoundsDatapoints(mNumberOfInBoundsDatapoints)
{}

template <class ImpurityWalker>
BestSplitpointsWalkingSortedStep<ImpurityWalker>::~BestSplitpointsWalkingSortedStep()
{}

template <class ImpurityWalker>
PipelineStepI* BestSplitpointsWalkingSortedStep<ImpurityWalker>::Clone() const
{
    BestSplitpointsWalkingSortedStep* clone = new BestSplitpointsWalkingSortedStep<ImpurityWalker>(*this);
    return clone;
}

template <class ImpurityWalker>
void BestSplitpointsWalkingSortedStep<ImpurityWalker>::ProcessStep(const BufferCollectionStack& readCollection,
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
    const typename ImpurityWalker::BufferTypes::Index numberOfFeatures =  mFeatureValueOrdering == FEATURES_BY_DATAPOINTS ? featureValues.GetM() : featureValues.GetN();

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

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::DatapointCounts>& childCounts
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::DatapointCounts> >(ChildCountsBufferId);
    childCounts.Resize(numberOfFeatures, 1, 2);

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous>& leftYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> >(LeftYsBufferId);
    leftYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous>& rightYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> >(RightYsBufferId);
    rightYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    for(int f=0; f<numberOfFeatures; f++)
    {
        impurityWalker.Reset();

        bool hasValidSplit = false;
        typename ImpurityWalker::BufferTypes::ImpurityValue bestImpurity = -std::numeric_limits<typename ImpurityWalker::BufferTypes::ImpurityValue>::max();
        typename ImpurityWalker::BufferTypes::FeatureValue bestSplitpoint = std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::min();
        typename ImpurityWalker::BufferTypes::SufficientStatsContinuous bestLeftChildCounts = typename ImpurityWalker::BufferTypes::SufficientStatsContinuous(0);
        typename ImpurityWalker::BufferTypes::SufficientStatsContinuous bestRightChildCounts = typename ImpurityWalker::BufferTypes::SufficientStatsContinuous(0);
        VectorBufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> bestLeftYs(impurityWalker.GetYDim());
        VectorBufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> bestRightYs(impurityWalker.GetYDim());

        FeatureSorter<typename ImpurityWalker::BufferTypes::FeatureValue> sorter(featureValues, mFeatureValueOrdering, f);
        sorter.Sort();

        typename ImpurityWalker::BufferTypes::FeatureValue boundsMin = std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::max();
        typename ImpurityWalker::BufferTypes::FeatureValue boundsMax = -std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::max();
        if( mNumberOfInBoundsDatapoints > 0)
        {
            const int numberOfSamples = sorter.GetNumberOfSamples();
            std::vector<int> inboundSamples(numberOfSamples);
            const int numberOfInBoundsSamples = std::min(numberOfSamples, mNumberOfInBoundsDatapoints);
            sampleWithOutReplacement(&inboundSamples[0], numberOfSamples, numberOfInBoundsSamples);

            for(int sortedIndex=0; sortedIndex<numberOfSamples; sortedIndex++)
            {
                if(inboundSamples[sortedIndex] )
                {
                    boundsMin = std::min(boundsMin, sorter.GetFeatureValue(sortedIndex));
                    boundsMax = std::max(boundsMax, sorter.GetFeatureValue(sortedIndex));
                }
            }
        }
        else
        {
            boundsMin = -std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::max();
            boundsMax = std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::max();
        }

        for(typename ImpurityWalker::BufferTypes::Index sortedIndex=0; sortedIndex<sorter.GetNumberOfSamples()-1; sortedIndex++)
        {
            const typename ImpurityWalker::BufferTypes::Index i = sorter.GetUnSortedIndex(sortedIndex);

            impurityWalker.MoveLeftToRight(i);

            const typename ImpurityWalker::BufferTypes::FeatureValue consecutiveFeatureDelta = sorter.GetFeatureValue(sortedIndex+1) - sorter.GetFeatureValue(sortedIndex);
            if((std::abs(consecutiveFeatureDelta) > std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::epsilon())
              && (sorter.GetFeatureValue(sortedIndex) >= boundsMin)
              && (sorter.GetFeatureValue(sortedIndex) <= boundsMax)
              && impurityWalker.Impurity() > bestImpurity)
            {
                hasValidSplit = true;
                bestImpurity = impurityWalker.Impurity();
                bestLeftChildCounts = impurityWalker.GetLeftChildCounts();
                bestRightChildCounts = impurityWalker.GetRightChildCounts();
                bestLeftYs = impurityWalker.GetLeftYs();
                bestRightYs = impurityWalker.GetRightYs();

                switch(mSplitpointLocation)
                {
                    case AT_MIDPOINT:
                        bestSplitpoint = sorter.GetFeatureValue(sortedIndex) + 0.5*consecutiveFeatureDelta;
                        break;
                    case AT_DATAPOINT:
                        bestSplitpoint = sorter.GetFeatureValue(sortedIndex);
                        break;
                    case UNIFORM_AT_GAP:
                        boost::uniform_real<> uniform_splitpoint(0.0, 1.0);
                        boost::variate_generator<boost::mt19937&,boost::uniform_real<> > var_uniform_splitpoint(gen, uniform_splitpoint);
                        bestSplitpoint = sorter.GetFeatureValue(sortedIndex) + var_uniform_splitpoint()*consecutiveFeatureDelta;
                        break;
                }
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