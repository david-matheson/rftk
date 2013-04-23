#pragma once

#include <vector>
#include <limits>  

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollectionStack.h"
#include "SplitSelectorBuffers.h"
#include "ShouldSplitCriteriaI.h"
#include "FinalizerI.h"
#include "SplitSelectorInfo.h"

// ----------------------------------------------------------------------------
//
// Loop through all feature/splitpoint pairs and select the best valid split
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class SplitSelector
{
public:
    SplitSelector(std::vector<SplitSelectorBuffers>& splitStatistics,
                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                  const FinalizerI<FloatType>* finalizer);
    ~SplitSelector();

    SplitSelectorInfo<FloatType, IntType> ProcessSplits(const BufferCollectionStack& bufferCollectionStack, int depth) const;

private:
    std::vector<SplitSelectorBuffers>& mSplitSelectorBuffers;
    const ShouldSplitCriteriaI* mShouldSplitCriteria;
    const FinalizerI<FloatType>* mFinalizer;
};

template <class FloatType, class IntType>
SplitSelector<FloatType, IntType>::SplitSelector( std::vector<SplitSelectorBuffers>& splitStatistics,
                                                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                                                  const FinalizerI<FloatType>* finalizer)
: mSplitSelectorBuffers(splitStatistics)
, mShouldSplitCriteria(shouldSplitCriteria->Clone())
, mFinalizer(finalizer->Clone())
{}

template <class FloatType, class IntType>
SplitSelector<FloatType, IntType>::~SplitSelector()
{
    mSplitSelectorBuffers.clear();
    delete mShouldSplitCriteria;
    delete mFinalizer;
}

template <class FloatType, class IntType>
SplitSelectorInfo<FloatType, IntType> SplitSelector<FloatType, IntType>::ProcessSplits(const BufferCollectionStack& readCollection, int depth) const
{
    FloatType maxImpurity = std::numeric_limits<FloatType>::min();
    int bestSplitSelectorBuffers = SPLIT_SELECTOR_NO_SPLIT;
    int bestFeature = SPLIT_SELECTOR_NO_SPLIT;
    int bestThreshold = SPLIT_SELECTOR_NO_SPLIT;

    for(int s=0; s<mSplitSelectorBuffers.size(); s++)
    {
        const SplitSelectorBuffers& ssb = mSplitSelectorBuffers[s];

        const MatrixBufferTemplate<FloatType>& impurities
           = readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(ssb.mImpurityBufferId);

        const VectorBufferTemplate<IntType>& splitpointCounts
               = readCollection.GetBuffer< VectorBufferTemplate<IntType> >(ssb.mSplitpointsCountsBufferId);

        const Tensor3BufferTemplate<FloatType>& childCounts
               = readCollection.GetBuffer< Tensor3BufferTemplate<FloatType> >(ssb.mChildCountsBufferId);

        for(int f=0; f<impurities.GetM(); f++)
        {
            for(int t=0; t<splitpointCounts.Get(f); t++)
            {
                const FloatType impurity = impurities.Get(f,t);
                const FloatType leftCounts = childCounts.Get(f,t,0);
                const FloatType rightCounts = childCounts.Get(f,t,1);
                if( impurity > maxImpurity 
                    && mShouldSplitCriteria->ShouldSplit(depth, impurity, leftCounts+rightCounts, leftCounts, rightCounts) )
                {
                    maxImpurity = impurity;
                    bestSplitSelectorBuffers = s;
                    bestFeature = f;
                    bestThreshold = t;
                }
            }
        } 
    }

    SplitSelectorInfo<FloatType, IntType> result(mSplitSelectorBuffers[bestSplitSelectorBuffers], 
                                                readCollection, mFinalizer, bestFeature, bestThreshold);
    return result;
}