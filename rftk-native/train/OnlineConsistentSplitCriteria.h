#pragma once

#include "MatrixBuffer.h"
#include "SplitCriteriaI.h"


class OnlineConsistentSplitCriteria : public SplitCriteriaI
{
public:
    OnlineConsistentSplitCriteria(  float growthRate,
                                    float minImpurity,
                                    float minNumberOfSamplesFirstSplit,
                                    float maxNumberOfSamplesFirstSplit);

    virtual ~OnlineConsistentSplitCriteria();

    virtual SPLT_CRITERIA ShouldSplit(   int treeDepth,
                                        const MatrixBufferFloat& impurityValues,
                                        const MatrixBufferFloat& childCounts) const;

    virtual int BestSplit(  int treeDepth,
                            const MatrixBufferFloat& impurityValues,
                            const MatrixBufferFloat& childCounts ) const;
private:
    const float mGrowthRate;
    const float mMinImpurity;
    const float mMinNumberOfSamplesFirstSplit;
    const float mMaxNumberOfSamplesFirstSplit;
};