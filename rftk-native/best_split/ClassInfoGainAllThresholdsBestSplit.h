#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "BestSplitI.h"


class ClassInfoGainAllThresholdsBestSplit : public BestSplitI {
public:
    ClassInfoGainAllThresholdsBestSplit(  float ratioOfThresholdsToTest,
                                        int minNumberThresholdsToTest,
                                        int numberOfClasses);
    virtual ~ClassInfoGainAllThresholdsBestSplit();

    virtual BestSplitI* Clone() const;

    virtual int GetYDim() const;

    virtual void BestSplits( const BufferCollection& data,
                            Float32MatrixBuffer& impurityOut,
                            Float32MatrixBuffer& thresholdOut,
                            Float32MatrixBuffer& childCountsOut,
                            Float32MatrixBuffer& leftYsOut,
                            Float32MatrixBuffer& rightYsOut) const;
private:
  float mRatioOfThresholdsToTest;
  int mMinNumberThresholdsToTest;
  int mNumberOfClasses;
};