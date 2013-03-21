#pragma once

#include "buffers/MatrixBuffer.h"
#include "buffers/BufferCollection.h"

class BestSplitI //Already exists
{
public:
    BestSplitI() {} //Needed by swig for pseudo abstract base classes
    virtual ~BestSplitI() {} //Needed by swig for pseudo abstract base classes

    virtual BestSplitI* Clone() const=0;

    virtual int GetYDim() const=0;

    virtual void BestSplits( const BufferCollection& data,
                            Float32VectorBuffer& impurityOut,
                            Float32VectorBuffer& thresholdOut,
                            Float32MatrixBuffer& childCountsOut,
                            Float32MatrixBuffer& leftYsOut,
                            Float32MatrixBuffer& rightYsOut) const=0;
};

