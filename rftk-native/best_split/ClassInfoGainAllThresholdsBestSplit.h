#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "BestSplitI.h"


class ClassInfoGainAllThresholdsBestSplit : public BestSplitI {
public:
    ClassInfoGainAllThresholdsBestSplit(  float ratioOfThresholdsToTest,
                                        int minNumberThresholdsToTest,
                                        int numberOfClasses);
    ~ClassInfoGainAllThresholdsBestSplit();

    virtual BestSplitI* Clone() const;

    virtual int GetYDim() const;

    virtual void BestSplits( BufferCollection& data,
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferFloat& childCountsOut,
                            MatrixBufferFloat& leftYsOut,
                            MatrixBufferFloat& rightYsOut) const;
private:
  float mRatioOfThresholdsToTest;
  int mMinNumberThresholdsToTest;
  int mNumberOfClasses;
};