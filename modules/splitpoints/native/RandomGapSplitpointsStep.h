#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>

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
// Finds a random gap in the feature values and then finds a random splitpoint
// in the random gap
//
// ----------------------------------------------------------------------------
template <class ImpurityWalker>
class RandomGapSplitpointsStep : public PipelineStepI
{
public:
    RandomGapSplitpointsStep (const ImpurityWalker& impurityWalker,
                              const BufferId& featureValues,
                              FeatureValueOrdering featureValueOrdering );
    virtual ~RandomGapSplitpointsStep();

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

};


template <class ImpurityWalker>
RandomGapSplitpointsStep<ImpurityWalker>::RandomGapSplitpointsStep(const ImpurityWalker& impurityWalker,
                                                                      const BufferId& featureValues,
                                                                      FeatureValueOrdering featureValueOrdering )
: PipelineStepI("RandomGapSplitpointsStep")
, ImpurityBufferId( GetBufferId("Impurity") )
, SplitpointBufferId( GetBufferId("Splitpoints") )
, SplitpointCountsBufferId( GetBufferId("SplitpointsCounts") )
, ChildCountsBufferId( GetBufferId("ChildCounts") )
, LeftYsBufferId( GetBufferId("LeftYs") )
, RightYsBufferId( GetBufferId("RightYs") )
, mImpurityWalker(impurityWalker)
, mFeatureValuesBufferId(featureValues)
, mFeatureValueOrdering(featureValueOrdering)
{}

template <class ImpurityWalker>
RandomGapSplitpointsStep<ImpurityWalker>::~RandomGapSplitpointsStep()
{}

template <class ImpurityWalker>
PipelineStepI* RandomGapSplitpointsStep<ImpurityWalker>::Clone() const
{
    RandomGapSplitpointsStep* clone = new RandomGapSplitpointsStep<ImpurityWalker>(*this);
    return clone;
}

template <class ImpurityWalker>
void RandomGapSplitpointsStep<ImpurityWalker>::ProcessStep(const BufferCollectionStack& readCollection,
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

    // Bind output buffers
    MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::ImpurityValue>& impurities
           = writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::ImpurityValue> >(ImpurityBufferId);
    impurities.Resize(numberOfFeatures,1);

    MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::FeatureValue>& splitpoints
           = writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename ImpurityWalker::BufferTypes::FeatureValue> >(SplitpointBufferId);
    splitpoints.Resize(numberOfFeatures,1);

    VectorBufferTemplate<typename ImpurityWalker::BufferTypes::Index>& splitpointsCounts
           = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename ImpurityWalker::BufferTypes::Index> >(SplitpointCountsBufferId);
    splitpointsCounts.Resize(numberOfFeatures);

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::DatapointCounts>& childCounts
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::DatapointCounts> >(ChildCountsBufferId);
    childCounts.Resize(numberOfFeatures, 1, 2);

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous>& leftYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> >(LeftYsBufferId);
    leftYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous>& rightYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> >(RightYsBufferId);
    rightYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    for(typename ImpurityWalker::BufferTypes::Index f=0; f<numberOfFeatures; f++)
    {

        impurityWalker.Reset();

        typename ImpurityWalker::BufferTypes::ImpurityValue bestImpurity = -std::numeric_limits<typename ImpurityWalker::BufferTypes::ImpurityValue>::max();
        typename ImpurityWalker::BufferTypes::FeatureValue bestSplitpoint = std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::min();
        typename ImpurityWalker::BufferTypes::SufficientStatsContinuous bestLeftChildCounts = typename ImpurityWalker::BufferTypes::SufficientStatsContinuous(0);
        typename ImpurityWalker::BufferTypes::SufficientStatsContinuous bestRightChildCounts = typename ImpurityWalker::BufferTypes::SufficientStatsContinuous(0);
        VectorBufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> bestLeftYs(impurityWalker.GetYDim());
        VectorBufferTemplate<typename ImpurityWalker::BufferTypes::SufficientStatsContinuous> bestRightYs(impurityWalker.GetYDim());

        FeatureSorter<typename ImpurityWalker::BufferTypes::FeatureValue> sorter(featureValues, mFeatureValueOrdering, f);
        sorter.Sort();
  
        ASSERT(sorter.GetNumberOfSamples()>=2)

        //count the number of valid gaps
        typename ImpurityWalker::BufferTypes::Index numberOfGaps = 0;
        for(typename ImpurityWalker::BufferTypes::Index sortedIndex=0; sortedIndex<sorter.GetNumberOfSamples()-1; sortedIndex++)
        {
            const typename ImpurityWalker::BufferTypes::FeatureValue consecutiveFeatureDelta = sorter.GetFeatureValue(sortedIndex+1) - sorter.GetFeatureValue(sortedIndex);
            if(std::abs(consecutiveFeatureDelta) > std::numeric_limits<const typename ImpurityWalker::BufferTypes::FeatureValue>::epsilon())
            {
                numberOfGaps++;
            }
        }

        if(numberOfGaps > 0)
        {
            // choose a gap uniformly at random
            boost::uniform_int<> uniform_gap(0, numberOfGaps-1);
            boost::variate_generator<boost::mt19937&,boost::uniform_int<> > var_uniform_gap(gen, uniform_gap);
            const int finalGapIndex = var_uniform_gap();
            int gapIndex = 0;
            for(int sortedIndex=0; sortedIndex<sorter.GetNumberOfSamples()-1; sortedIndex++)
            {
                const int i = sorter.GetUnSortedIndex(sortedIndex);
                impurityWalker.MoveLeftToRight(i);

                const typename ImpurityWalker::BufferTypes::FeatureValue consecutiveFeatureDelta = sorter.GetFeatureValue(sortedIndex+1) - sorter.GetFeatureValue(sortedIndex);
                if(std::abs(consecutiveFeatureDelta) > std::numeric_limits<typename ImpurityWalker::BufferTypes::FeatureValue>::epsilon())
                {
                    if( gapIndex >= finalGapIndex )
                    {
                        // choose a split uniformly at random in the gap
                        boost::uniform_real<> uniform_split(sorter.GetFeatureValue(sortedIndex), sorter.GetFeatureValue(sortedIndex+1));
                        boost::variate_generator<boost::mt19937&,boost::uniform_real<> > var_uniform_split(gen, uniform_split);
                        bestImpurity = impurityWalker.Impurity();
                        bestSplitpoint = var_uniform_split();
                        bestLeftChildCounts = impurityWalker.GetLeftChildCounts();
                        bestRightChildCounts = impurityWalker.GetRightChildCounts();
                        bestLeftYs = impurityWalker.GetLeftYs();
                        bestRightYs = impurityWalker.GetRightYs();
                        break;
                    }
                    gapIndex++;
                }
            }
        }


        impurities.Set(f, 0, std::max(typename ImpurityWalker::BufferTypes::ImpurityValue(0), bestImpurity));
        splitpoints.Set(f, 0, bestSplitpoint);
        splitpointsCounts.Set(f, 1);
        childCounts.Set(f, 0, 0, bestLeftChildCounts);
        childCounts.Set(f, 0, 1, bestRightChildCounts);
        leftYs.SetRow(f, 0, bestLeftYs );
        rightYs.SetRow(f, 0, bestRightYs );
    }
}