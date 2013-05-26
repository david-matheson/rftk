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
#include "SplitSelectorI.h"

// ----------------------------------------------------------------------------
//
// Find the best split (by impurity) and then check if it is valid
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class WaitForBestSplitSelector: public SplitSelectorI<FloatType, IntType>
{
public:
    WaitForBestSplitSelector(const std::vector<SplitSelectorBuffers>& splitBuffers,
                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                  const FinalizerI<FloatType>* finalizer);

    WaitForBestSplitSelector(const std::vector<SplitSelectorBuffers>& splitBuffers,
                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                  const FinalizerI<FloatType>* finalizer,
                  const SplitBuffersI* bufferSplitter);

    virtual ~WaitForBestSplitSelector();

    virtual SplitSelectorInfo<FloatType, IntType> ProcessSplits(const BufferCollectionStack& bufferCollectionStack, int depth) const;

    virtual SplitSelectorI<FloatType, IntType>* Clone() const;

private:
    std::vector<SplitSelectorBuffers> mSplitSelectorBuffers;
    const ShouldSplitCriteriaI* mShouldSplitCriteria;
    const FinalizerI<FloatType>* mFinalizer;
    const SplitBuffersI* mBufferSplitter;
};

template <class FloatType, class IntType>
WaitForBestSplitSelector<FloatType, IntType>::WaitForBestSplitSelector( const std::vector<SplitSelectorBuffers>& splitBuffers,
                                                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                                                  const FinalizerI<FloatType>* finalizer)
: mSplitSelectorBuffers(splitBuffers)
, mShouldSplitCriteria(shouldSplitCriteria->Clone())
, mFinalizer(finalizer->Clone())
, mBufferSplitter(NULL)
{}

template <class FloatType, class IntType>
WaitForBestSplitSelector<FloatType, IntType>::WaitForBestSplitSelector( const std::vector<SplitSelectorBuffers>& splitBuffers,
                                                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                                                  const FinalizerI<FloatType>* finalizer,
                                                  const SplitBuffersI* bufferSplitter)
: mSplitSelectorBuffers(splitBuffers)
, mShouldSplitCriteria(shouldSplitCriteria->Clone())
, mFinalizer(finalizer->Clone())
, mBufferSplitter((bufferSplitter != NULL) ? bufferSplitter->Clone() : NULL)
{}

template <class FloatType, class IntType>
WaitForBestSplitSelector<FloatType, IntType>::~WaitForBestSplitSelector()
{
    mSplitSelectorBuffers.clear();
    delete mShouldSplitCriteria;
    delete mBufferSplitter;
    delete mFinalizer;
}


template <class FloatType, class IntType>
SplitSelectorInfo<FloatType, IntType> WaitForBestSplitSelector<FloatType, IntType>::ProcessSplits(const BufferCollectionStack& readCollection, int depth) const
{
    FloatType maxImpurity = std::numeric_limits<FloatType>::min();
    int bestWaitForBestmSplitSelectorBuffers = SPLIT_SELECTOR_NO_SPLIT;
    int bestFeature = SPLIT_SELECTOR_NO_SPLIT;
    int bestSplitpoint = SPLIT_SELECTOR_NO_SPLIT;

    const int numberOfWaitForBestmSplitSelectorBuffers = mSplitSelectorBuffers.size();

    for(int s=0; s<numberOfWaitForBestmSplitSelectorBuffers; s++)
    {
        const SplitSelectorBuffers& ssb = mSplitSelectorBuffers[s];

        const MatrixBufferTemplate<FloatType>& impurities
           = readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(ssb.mImpurityBufferId);

        const VectorBufferTemplate<IntType>& splitpointCounts
               = readCollection.GetBuffer< VectorBufferTemplate<IntType> >(ssb.mSplitpointsCountsBufferId);

        for(int f=0; f<impurities.GetM(); f++)
        {
            for(int t=0; t<splitpointCounts.Get(f); t++)
            {
                const FloatType impurity = impurities.Get(f,t);
                if( impurity > maxImpurity )
                {
                    maxImpurity = impurity;
                    bestWaitForBestmSplitSelectorBuffers = s;
                    bestFeature = f;
                    bestSplitpoint = t;
                }
            }
        }
    }

    const bool isSet = bestWaitForBestmSplitSelectorBuffers !=  SPLIT_SELECTOR_NO_SPLIT
                        && bestFeature != SPLIT_SELECTOR_NO_SPLIT
                        && bestSplitpoint != SPLIT_SELECTOR_NO_SPLIT;

    if( isSet )
    {
        const SplitSelectorBuffers& ssb = mSplitSelectorBuffers[bestWaitForBestmSplitSelectorBuffers];
        const Tensor3BufferTemplate<FloatType>& childCounts
                    = readCollection.GetBuffer< Tensor3BufferTemplate<FloatType> >(ssb.mChildCountsBufferId);

        // Reset if should not split
        const FloatType leftCounts = childCounts.Get(bestFeature,bestSplitpoint,LEFT_CHILD_INDEX);
        const FloatType rightCounts = childCounts.Get(bestFeature,bestSplitpoint,RIGHT_CHILD_INDEX);
        if( !mShouldSplitCriteria->ShouldSplit(depth, maxImpurity, leftCounts+rightCounts, leftCounts, rightCounts))
        {
            bestWaitForBestmSplitSelectorBuffers = SPLIT_SELECTOR_NO_SPLIT;
            bestFeature = SPLIT_SELECTOR_NO_SPLIT;
            bestSplitpoint = SPLIT_SELECTOR_NO_SPLIT;
        }
    }

    return SplitSelectorInfo<FloatType, IntType>(mSplitSelectorBuffers[bestWaitForBestmSplitSelectorBuffers],
                                            readCollection, mFinalizer, mBufferSplitter,
                                            bestFeature, bestSplitpoint, depth);
}

template <class FloatType, class IntType>
SplitSelectorI<FloatType, IntType>* WaitForBestSplitSelector<FloatType, IntType>::Clone() const
{
    WaitForBestSplitSelector* clone = new WaitForBestSplitSelector<FloatType, IntType>(mSplitSelectorBuffers, 
                                                                                      mShouldSplitCriteria, 
                                                                                      mFinalizer, 
                                                                                      mBufferSplitter);
    return clone;
}