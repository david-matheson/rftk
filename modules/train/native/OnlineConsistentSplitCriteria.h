#pragma once

#include <VectorBuffer.h>
#include <MatrixBuffer.h>
#include "SplitCriteriaI.h"


class OnlineConsistentSplitCriteria : public SplitCriteriaI
{
public:
    OnlineConsistentSplitCriteria(  float growthRate,
                                    float minImpurity,
                                    float minNumberOfSamplesFirstSplit,
                                    float maxNumberOfSamplesFirstSplit,
                                    int maxDepth);
    virtual ~OnlineConsistentSplitCriteria();
    virtual SplitCriteriaI* Clone() const;

    virtual bool ShouldProcessNode( int treeDepth ) const;

    virtual SPLT_CRITERIA ShouldSplit(   int treeDepth,
                                        const Float32VectorBuffer& impurityValues,
                                        const Float32MatrixBuffer& childCounts) const;

    virtual int BestSplit(  int treeDepth,
                            const Float32VectorBuffer& impurityValues,
                            const Float32MatrixBuffer& childCounts ) const;

    virtual int MinTotalSamples( int treeDepth ) const;
private:
    const float mGrowthRate;
    const float mMinImpurity;
    const float mMinNumberOfSamplesFirstSplit;
    const float mMaxNumberOfSamplesFirstSplit;
    const int mMaxDepth;
};