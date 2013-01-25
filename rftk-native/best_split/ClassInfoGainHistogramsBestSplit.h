#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "BestSplitI.h"


class ClassInfoGainHistogramsBestSplit : public BestSplitI {
public:
    ClassInfoGainHistogramsBestSplit(int numberOfClasses);
    ~ClassInfoGainHistogramsBestSplit();

    virtual BestSplitI* Clone() const;

    virtual int GetYDim() const;

    virtual void BestSplits( const BufferCollection& data,
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferFloat& childCountsOut,
                            MatrixBufferFloat& leftYsOut,
                            MatrixBufferFloat& rightYsOut) const;

private:
  int mNumberOfClasses;
};