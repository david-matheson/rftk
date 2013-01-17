#pragma once

#include "MatrixBuffer.h"
#include "SplitCriteriaI.h"


class OnlineAlphaBetaSplitCriteria : public SplitCriteriaI
{
public:
    OnlineAlphaBetaSplitCriteria(   int maxDepth, 
                                    float minImpurity, 
                                    float minNumberOfSamples);

    virtual ~OnlineAlphaBetaSplitCriteria();

    virtual SPLT_CRITERIA ShouldSplit(   int treeDepth,
                                        const MatrixBufferFloat& impurityValues,
                                        const MatrixBufferFloat& childCounts) const;

    virtual int BestSplit(  int treeDepth,
                            const MatrixBufferFloat& impurityValues,
                            const MatrixBufferFloat& childCounts ) const;
private:
    const int mMaxDepth;
    const float mMinImpurity;
    const float mMinNumberOfSamples;
};