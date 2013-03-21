#pragma once

#include "buffers/MatrixBuffer.h"
#include "SplitCriteriaI.h"


class OfflineSplitCriteria : public SplitCriteriaI
{
public:
    OfflineSplitCriteria(   int maxDepth,
                            float minImpurity,
                            float minSampleCounts,
                            float minChildSampleCounts);

    virtual ~OfflineSplitCriteria();
    virtual SplitCriteriaI* Clone() const;

    virtual SPLT_CRITERIA ShouldSplit(   int treeDepth,
                                        const Float32VectorBuffer& impurityValues,
                                        const Float32MatrixBuffer& childCounts) const;

    virtual int BestSplit(  int treeDepth,
                            const Float32VectorBuffer& impurityValues,
                            const Float32MatrixBuffer& childCounts ) const;

    virtual int MinTotalSamples( int treeDepth ) const;
private:
    const int mMaxDepth;
    const float mMinImpurity;
    const float mMinSampleCounts;
    const float mMinChildSampleCounts;
};