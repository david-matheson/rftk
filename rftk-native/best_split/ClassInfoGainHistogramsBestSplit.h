#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "BestSplitI.h"


class ClassInfoGainHistogramsBestSplit : public BestSplitI {
public:
    ClassInfoGainHistogramsBestSplit(int maxClass);

    ~ClassInfoGainHistogramsBestSplit();

    virtual int GetYDim() const;

    virtual void BestSplits( BufferCollection& data,
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferFloat& childCountsOut,
                            MatrixBufferFloat& leftYsOut,
                            MatrixBufferFloat& rightYsOut) const;

private:
  int mMaxClass;
};