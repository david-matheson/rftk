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
    virtual SplitCriteriaI* Clone() const;

    virtual SPLT_CRITERIA ShouldSplit(   int treeDepth,
                                        const Float32MatrixBuffer& impurityValues,
                                        const Float32MatrixBuffer& childCounts) const;

    virtual int BestSplit(  int treeDepth,
                            const Float32MatrixBuffer& impurityValues,
                            const Float32MatrixBuffer& childCounts ) const;
private:
    const int mMaxDepth;
    const float mMinImpurity;
    const float mMinNumberOfSamples;
};