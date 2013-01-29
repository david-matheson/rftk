#pragma once

#include "VectorBuffer.h"
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
    SplitCriteriaI() {}     //Needed by swig for pseudo abstract base classes
    virtual ~SplitCriteriaI() {}    //Needed by swig for pseudo abstract base classes
    virtual SplitCriteriaI* Clone() const=0;

    virtual SPLT_CRITERIA ShouldSplit(   int treeDepth,
                                        const Float32VectorBuffer& impurityValues,
                                        const Float32MatrixBuffer& childCounts) const=0;

    virtual int BestSplit(  int treeDepth,
                            const Float32VectorBuffer& impurityValues,
                            const Float32MatrixBuffer& childCounts ) const=0;
};