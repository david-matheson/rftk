#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "BestSplitI.h"


class ClassInfoGainAllThresholdsBestSplit : public BestSplitI {
public:
    ClassInfoGainAllThresholdsBestSplit(  float ratioOfThresholdsToTest,
                                        int minNumberThresholdsToTest,
                                        int maxClass);

    ~ClassInfoGainAllThresholdsBestSplit();

    virtual void BestSplits( BufferCollection& data,
                            // const MatrixBufferInt& sampleIndices,
                            // const MatrixBufferFloat& featureValues, // contained in data (if needed)
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferFloat& childCountsOut,
                            MatrixBufferFloat& leftYsOut,
                            MatrixBufferFloat& rightYsOut) const;

    // virtual void BestSplits(  BufferCollection& data,  //must contain "ClassLabels" and "SampleWeights"
    //                         const MatrixBufferInt& sampleIndices,
    //                         const MatrixBufferFloat& featureValues,
    //                         MatrixBufferFloat& impurityOut,
    //                         MatrixBufferFloat& thresholdOut);

private:
  float mRatioOfThresholdsToTest;
  int mMinNumberThresholdsToTest;
  int mMaxClass;
};