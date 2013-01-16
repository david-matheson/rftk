#pragma once

#include "MatrixBuffer.h"

enum SPLT_CRITERIA
{
    SPLT_CRITERIA_MORE_DATA_REQUIRED,
    SPLT_CRITERIA_READY_TO_SPLIT,
    SPLT_CRITERIA_STOP,
};

class SplitCriteriaI
{
public:
    virtual ~SplitCriteriaI() {}

    virtual SPLT_CRITERIA ShouldSplit(   int treeDepth,
                                        const MatrixBufferFloat& impurityValues,
                                        const MatrixBufferFloat& childCounts) const { return SPLT_CRITERIA_MORE_DATA_REQUIRED; }

    virtual int BestSplit(  int treeDepth,
                            const MatrixBufferFloat& impurityValues,
                            const MatrixBufferFloat& childCounts ) const { return -1;}
};