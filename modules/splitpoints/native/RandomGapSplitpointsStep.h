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
                                boost::mt19937& gen) const;

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
: ImpurityBufferId( GetBufferId("Impurity") )
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

    MatrixBufferTemplate<typename ImpurityWalker::Float>& splitpoints
           = writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename ImpurityWalker::Float> >(SplitpointBufferId);
    splitpoints.Resize(numberOfFeatures,1);

    VectorBufferTemplate<typename ImpurityWalker::Int>& splitpointsCounts
           = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename ImpurityWalker::Int> >(SplitpointCountsBufferId);
    splitpointsCounts.Resize(numberOfFeatures);

    Tensor3BufferTemplate<typename ImpurityWalker::Float>& childCounts
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::Float> >(ChildCountsBufferId);
    childCounts.Resize(numberOfFeatures, 1, 2);

    Tensor3BufferTemplate<typename ImpurityWalker::Float>& leftYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::Float> >(LeftYsBufferId);
    leftYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    Tensor3BufferTemplate<typename ImpurityWalker::Float>& rightYs
           = writeCollection.GetOrAddBuffer< Tensor3BufferTemplate<typename ImpurityWalker::Float> >(RightYsBufferId);
    rightYs.Resize(numberOfFeatures, 1, impurityWalker.GetYDim());

    for(int f=0; f<numberOfFeatures; f++)
    {

        impurityWalker.Reset();

        typename ImpurityWalker::Float bestImpurity = std::numeric_limits<typename ImpurityWalker::Float>::min();
        typename ImpurityWalker::Float bestSplitpoint = std::numeric_limits<typename ImpurityWalker::Float>::min();
        typename ImpurityWalker::Float bestLeftChildCounts = typename ImpurityWalker::Float(0);
        typename ImpurityWalker::Float bestRightChildCounts = typename ImpurityWalker::Float(0);
        VectorBufferTemplate<typename ImpurityWalker::Float> bestLeftYs(impurityWalker.GetYDim());
        VectorBufferTemplate<typename ImpurityWalker::Float> bestRightYs(impurityWalker.GetYDim());

        FeatureSorter<typename ImpurityWalker::Float> sorter(featureValues, mFeatureValueOrdering, f);
        sorter.Sort();

        ASSERT(sorter.GetNumberOfSamples()>=2)

        //count the number of valid gaps
        int numberOfGaps = 0;
        for(int sortedIndex=0; sortedIndex<sorter.GetNumberOfSamples()-1; sortedIndex++)
        {
            const typename ImpurityWalker::Float consecutiveFeatureDelta = sorter.GetFeatureValue(sortedIndex+1) - sorter.GetFeatureValue(sortedIndex);
            if(std::abs(consecutiveFeatureDelta) > std::numeric_limits<typename ImpurityWalker::Float>::epsilon())
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

                const typename ImpurityWalker::Float consecutiveFeatureDelta = sorter.GetFeatureValue(sortedIndex+1) - sorter.GetFeatureValue(sortedIndex);
                if(std::abs(consecutiveFeatureDelta) > std::numeric_limits<typename ImpurityWalker::Float>::epsilon())
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


        impurities.Set(f, 0, bestImpurity);
        splitpoints.Set(f, 0, bestSplitpoint);
        splitpointsCounts.Set(f, 1);
        childCounts.Set(f, 0, 0, bestLeftChildCounts);
        childCounts.Set(f, 0, 1, bestRightChildCounts);
        leftYs.SetRow(f, 0, bestLeftYs );
        rightYs.SetRow(f, 0, bestRightYs );
    }
}