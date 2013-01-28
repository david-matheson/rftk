#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

class BestSplitI //Already exists
{
public:
    BestSplitI() {} //Needed by swig for pseudo abstract base classes
    virtual ~BestSplitI() {} //Needed by swig for pseudo abstract base classes

    virtual BestSplitI* Clone() const=0;

    virtual int GetYDim() const=0;

    virtual void BestSplits( const BufferCollection& data,
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferFloat& childCountsOut,
                            MatrixBufferFloat& leftYsOut,
                            MatrixBufferFloat& rightYsOut) const=0;
};

