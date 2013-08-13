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
#include "SplitBuffersI.h"


// ----------------------------------------------------------------------------
//
// Loop through all feature/splitpoint pairs and select the best valid split
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class SplitSelector: public SplitSelectorI<BufferTypes>
{
public:
    SplitSelector(const std::vector<SplitSelectorBuffers>& splitBuffers,
                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                  const FinalizerI<BufferTypes>* finalizer);

    SplitSelector(const std::vector<SplitSelectorBuffers>& splitBuffers,
                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                  const FinalizerI<BufferTypes>* finalizer,
                  const SplitBuffersI* bufferSplitter);
    virtual ~SplitSelector();

    virtual SplitSelectorInfo<BufferTypes> ProcessSplits(const BufferCollectionStack& bufferCollectionStack, int depth) const;

    virtual SplitSelectorI<BufferTypes>* Clone() const;

private:
    std::vector<SplitSelectorBuffers> mSplitSelectorBuffers;
    const ShouldSplitCriteriaI* mShouldSplitCriteria;
    const FinalizerI<BufferTypes>* mFinalizer;
    const SplitBuffersI* mBufferSplitter;
};

template <class BufferTypes>
SplitSelector<BufferTypes>::SplitSelector( const std::vector<SplitSelectorBuffers>& splitBuffers,
                                                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                                                  const FinalizerI<BufferTypes>* finalizer)
: mSplitSelectorBuffers(splitBuffers)
, mShouldSplitCriteria(shouldSplitCriteria->Clone())
, mFinalizer(finalizer->Clone())
, mBufferSplitter(NULL)
{}

template <class BufferTypes>
SplitSelector<BufferTypes>::SplitSelector( const std::vector<SplitSelectorBuffers>& splitBuffers,
                                                  const ShouldSplitCriteriaI* shouldSplitCriteria,
                                                  const FinalizerI<BufferTypes>* finalizer,
                                                  const SplitBuffersI* bufferSplitter)
: mSplitSelectorBuffers(splitBuffers)
, mShouldSplitCriteria(shouldSplitCriteria->Clone())
, mFinalizer(finalizer->Clone())
, mBufferSplitter((bufferSplitter != NULL) ? bufferSplitter->Clone() : NULL)
{}

template <class BufferTypes>
SplitSelector<BufferTypes>::~SplitSelector()
{
    mSplitSelectorBuffers.clear();
    delete mShouldSplitCriteria;
    delete mBufferSplitter;
    delete mFinalizer;
}

template <class BufferTypes>
SplitSelectorInfo<BufferTypes> SplitSelector<BufferTypes>::ProcessSplits(const BufferCollectionStack& readCollection, int depth) const
{
    typename BufferTypes::ImpurityValue maxImpurity = -std::numeric_limits<typename BufferTypes::ImpurityValue>::max();
    int bestSplitSelectorBuffers = SPLIT_SELECTOR_NO_SPLIT;
    int bestFeature = SPLIT_SELECTOR_NO_SPLIT;
    int bestThreshold = SPLIT_SELECTOR_NO_SPLIT;

    for(unsigned int s=0; s<mSplitSelectorBuffers.size(); s++)
    {
        const SplitSelectorBuffers& ssb = mSplitSelectorBuffers[s];

        const MatrixBufferTemplate<typename BufferTypes::ImpurityValue>& impurities
           = readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ImpurityValue> >(ssb.mImpurityBufferId);

        const VectorBufferTemplate<typename BufferTypes::Index>& splitpointCounts
               = readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(ssb.mSplitpointsCountsBufferId);

        const Tensor3BufferTemplate<typename BufferTypes::DatapointCounts>& childCounts
               = readCollection.GetBuffer< Tensor3BufferTemplate<typename BufferTypes::DatapointCounts> >(ssb.mChildCountsBufferId);

        for(int f=0; f<impurities.GetM(); f++)
        {
            for(int t=0; t<splitpointCounts.Get(f); t++)
            {
                const typename BufferTypes::ImpurityValue impurity = impurities.Get(f,t);
                const typename BufferTypes::DatapointCounts leftCounts = childCounts.Get(f,t,0);
                const typename BufferTypes::DatapointCounts rightCounts = childCounts.Get(f,t,1);
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

    return SplitSelectorInfo<BufferTypes>(mSplitSelectorBuffers[bestSplitSelectorBuffers],
                                            readCollection, mFinalizer, mBufferSplitter,
                                            bestFeature, bestThreshold, depth);;
}

template <class BufferTypes>
SplitSelectorI<BufferTypes>* SplitSelector<BufferTypes>::Clone() const
{
    SplitSelector* clone = new SplitSelector<BufferTypes>(mSplitSelectorBuffers, mShouldSplitCriteria, mFinalizer, mBufferSplitter);
    return clone;
}